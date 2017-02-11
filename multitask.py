#!/usr/bin/python

''' main file for training word embeddings and get sentence embeddings	'''

from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import collections
import math
import os
import random
import zipfile
import glob
import ntpath
import re
import random
from itertools import compress
import cPickle
from pathlib import Path

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

from sklearn.base import BaseEstimator, TransformerMixin


DATA = "./data/"
DATA_BI = DATA+"parallel/"
DATA_MONO = DATA+"mono/"

DATA_TASK = DATA+"task/"
DATA_PROCESSED = DATA +"processed/"
#logs_path = './tmp/tensorflow_logs/new/'

# initialize data_indexes
data_mono_index = 0
data_bi_index = 0
data_task_mlp_index = 0

# Utility Functions

def preprocess_text(text):
	
	text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
	text = re.sub(r'[^\w\s]','',text) # remove all punctuations
	text = re.sub(' +',' ',text) # remove multiple spaces
	text = text.lower()
	text = text.strip() # remove ending space

	return text

def _attn_mul_fun(keys, query):
  return math_ops.reduce_sum(keys * query, [2])

def pad(l, content, width):
	l.extend([content] * (width - len(l)))
	return l

def read_data(lang_ext=1):
	"""Extract the first file enclosed in a zip file as a list of words"""

	data_mono = []
	mono_files = glob.glob(DATA_MONO+"*")
	for filename in mono_files:
		print(filename)
		f = open(filename,'r')
		ext = ":ID:" + filename.split(".")[-1]
		for line in f:
			if line != '':
				line = preprocess_text(line)
				# lang_ext is sticked to each token
				if lang_ext == 1:
					tokens = [x + ext for x in line.split()]
				else:
					tokens = line.split()
				data_mono.append(tokens)

	data_bi = []
	bi_files = glob.glob(DATA_BI+"*")
	for filename in bi_files:
		print(filename)
		sentence_pair_file = open(filename,'r')
		src_lang = ":ID:" + filename.split(".")[-1].split("-")[0]
		tgt_lang = ":ID:" + filename.split(".")[-1].split("-")[1]
		for sentence_pair_line in sentence_pair_file:
			sentence_pair_line = sentence_pair_line.rstrip()
			source_line, target_line = sentence_pair_line.split(" ||| ")

			source_line = preprocess_text(source_line)
			target_line = preprocess_text(target_line)

			if source_line != '' and target_line != '':
				source_tokens, target_tokens = source_line.split(' '), target_line.split(' ')
				if lang_ext == 1:
					source_tokens = [x + src_lang for x in source_tokens]
					target_tokens = [x + tgt_lang for x in target_tokens]
#			source_tokens.extend(['<eos>'])
#			target_tokens.extend(['<eos>'])
				data_bi.append([source_tokens,target_tokens])

	print(len(data_mono),len(data_bi))
	return data_mono,data_bi

# mlp : multilingual polarity
def read_dataset_task_mlp(dictionary,filename):
	data = {}
	num_classes = 3
	class_counter = 0
	classes = {}
#	classes_index = [0]*num_classes
	task_files = glob.glob(DATA_TASK+"multi-3-way-polarity/*")
	print(" Reading data for", "multi-3-way-polarity","prediction task")
	for file in task_files:
		print(file)
		read_file = open(file,'r')
		lang_ext = file.split("_")[-1].split(".")[0]
		data[lang_ext] = {}
		task_data = []
		for line in read_file:
			line = preprocess_text(line)
			line = line.split("\t")
			classes_index = [0]*num_classes
			if line[1] not in classes:
				classes[line[1]] = class_counter
				class_counter = class_counter + 1
			classes_index[classes[line[1]]] = 1
			task_entry = []
			line_ID = []
			if len(line) > 2:
				line = line[2].split()
				for word in line:
					word = word+":ID:"+lang_ext
					ID = dictionary.get(word)
					if ID is not None:
						line_ID.append(ID)
				if line_ID != []:
					task_entry.append(line_ID)
					task_entry.append(classes_index)
					task_data.append(task_entry)
	
		random.shuffle(task_data)

		# for each language divide data into train, dev & test
		data[lang_ext]['train'] = task_data[:int(len(task_data)*0.90)]
		data[lang_ext]['dev'] = task_data[int(len(task_data)*0.90):int(len(task_data)*0.95)]
		data[lang_ext]['test'] = task_data[int(len(task_data)*0.95):]

	cPickle.dump(data, open(filename, 'wb'))

def build_dataset(words, vocabulary_size = 200000):
	'''
	Build the dictionary and replace rare words with UNK token.
	
	Parameters
	----------
	words: list of tokens
	vocabulary_size: maximum number of top occurring tokens to produce, 
		rare tokens will be replaced by 'UNK'
	'''
	print("Build Dataset and dictionaries")

#	mono = open(DATA_PROCESSED+"mono.p")
#	bi = open(DATA_PROCESSED+"bi.p")

	seq_mono = words[0]
	seq_bi = words[1]
	words_total = []


	for seq in seq_mono:
		words_total.extend(seq)
	for sent_pair in seq_bi:
		for seq in sent_pair:
			words_total.extend(seq)

	del words # saving memeory

	# create a dictionary counnter
	count = [['UNK', -1]]
	count.extend(collections.Counter(words_total).most_common(vocabulary_size - 1))

	del words_total # saving memory

	dictionary = dict() # {word : index}
	data_mono = list()
	data_bi = list()
	for word, _ in count:
		word = word
		dictionary[word] = len(dictionary)
		unk_count = 0
	
	## get mono-data indexed
	for seq in seq_mono:
		for i in range(0,len(seq)):
			if seq[i] in dictionary:
				index = dictionary[seq[i]]
			else:
				index = 0 # dictionary['UNK']
				unk_count += 1
			seq[i] = index
		data_mono.append(seq)

	cPickle.dump(data_mono, open(DATA_PROCESSED + 'mono.p', 'wb'))

	del data_mono # saving memory 
	## get bi-data indexed
	for sent_pair in seq_bi:
		pair = []
		for seq in sent_pair:
			for i in range(0,len(seq)):
				if seq[i] in dictionary:
					index = dictionary[seq[i]]
				else:
					index = 0 # dictionary['UNK']
					unk_count += 1
				seq[i] = index
			pair.append(seq)
		data_bi.append(pair)

	cPickle.dump(data_bi, open(DATA_PROCESSED + 'bi.p', 'wb'))
	
	del data_bi # saving memory	

	count[0][1] = unk_count # list of tuples (word,count)
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	cPickle.dump(dictionary, open(DATA_PROCESSED + 'dictionary.p', 'wb'))
	cPickle.dump(reverse_dictionary, open(DATA_PROCESSED + 'reverse_dictionary.p', 'wb'))
	cPickle.dump(count, open(DATA_PROCESSED + 'count.p', 'wb'))

def generate_batch_mono_skip(data_mono, batch_size, skip_window):
	'''
	generates a batch which means source and target list indexes
	'''
	global data_mono_index
	batch = []
	labels = []
	for i in range(batch_size):
		sent = data_mono[data_mono_index]
		for j in range(skip_window):		
			sent = ['<eos>'] + sent + ['<eos>']
		for j in range(skip_window,len(sent)-skip_window):
			for skip in range(1,skip_window+1):
				if sent[j-skip] != '<eos>': 
					batch.append(sent[j])
					labels.append(sent[j-skip])
				if sent[j+skip] != '<eos>':
					batch.append(sent[j])
					labels.append(sent[j+skip])
		data_mono_index = (data_mono_index + 1) % len(data_mono)
		if data_mono_index == 0:
			print("shuffle mono data")
			random.shuffle(data_mono)

	labels_final = np.ndarray(shape=(len(labels), 1), dtype=np.int32)
	batch_final = np.ndarray(shape=(len(batch)), dtype=np.int32)
	for i in range(len(labels)):
		labels_final[i,0] = labels[i]
	for i in range(len(batch)):
		batch_final[i] = batch[i]
	return batch_final, labels_final

def generate_batch_multi_skip(data_bi, multi_batch_size, window=5):
	'''
	1. generates a batch from source to target and target to source
	2. Predict all target words given a source word
	'''
	global data_bi_index
	batch = []
	labels = []
	for i in range(multi_batch_size):
		sent1 = data_bi[data_bi_index][0]
		sent2 = data_bi[data_bi_index][1]
		sent1_len = float(len(sent1))
		sent2_len = float(len(sent2))
		for j in range(len(sent1)):
			alignment = int((j/sent1_len) * sent2_len)
			window_high = alignment + window
			window_low = alignment - window
			if window_low < 0:
				window_low = 0

			for k in sent2[alignment:window_high]:
				# l1 -> l2
				batch.append(sent1[j])
				labels.append(k)

				# l2 -> l1
				batch.append(k)
				labels.append(sent1[j])

			for k in sent2[window_low:alignment]:
				# l1 -> l2
				batch.append(sent1[j])
				labels.append(k)

				# l2 -> l1
				batch.append(k)
				labels.append(sent1[j])

		data_bi_index = (data_bi_index + 1) % len(data_bi)
		if data_bi_index == 0:
			print("shuffle multi-skip data")
			random.shuffle(data_bi)
	batch = np.array(batch)
	labels = np.array(labels)

	labels_final = np.ndarray(shape=(len(labels), 1), dtype=np.int32)
	batch_final = np.ndarray(shape=(len(batch)), dtype=np.int32)
	for i in range(len(labels)):
		labels_final[i,0] = labels[i]
	for i in range(len(batch)):
		batch_final[i] = batch[i]
	return batch_final, labels_final

def generate_batch_task_mlp(data_task_mlp_train, task_batch_size, sen_length):
	'''
	generates CBOW batch of the task
	'''
	global data_task_mlp_index
#	sen_length = 20
	batch = []
	labels = []
	batch = np.ndarray(shape=(task_batch_size,sen_length), dtype=np.int32)
	labels = np.ndarray(shape=(task_batch_size, 3), dtype=np.int32)
	for i in range(task_batch_size):
		sent = data_task_mlp_train[data_task_mlp_index]
		sent[0] = pad(sent[0][:sen_length], 0, sen_length)
#		print(sent[0])
		batch[i,:] = sent[0]
		labels[i,:] = sent[1]
		data_task_mlp_index = (data_task_mlp_index + 1) % len(data_task_mlp_train)
		if data_task_mlp_index == 0:
			print("shuffle task_mlp train data")
			random.shuffle(data_task_mlp_train)

	return batch, labels



class MultiTask(BaseEstimator, TransformerMixin):

	def __init__(self, vocabulary_size=150000, embedding_size=200, batch_size=5,
		multi_batch_size=5, task_batch_size=20, skip_window=2, num_sampled=64,
		valid_size=16, valid_window=500, skip_gram_learning_rate=0.01, sen_length=20,
		task_mlp_learning_rate=0.1,num_steps=1400001, task_mlp_start=20000, 
		task_mlp_hidden=50, attention='true', attention_size = 150, task_tune='false', 
		joint='true', logs_path='./tmp/tensorflow_logs/test', num_threads=10,num_classes=3):

		# set parameters
		self.vocabulary_size = vocabulary_size # size of vocabulary
		self.embedding_size = embedding_size # Dimension of the embedding vectorself.
		self.batch_size = batch_size # mono-lingual batch size
		self.multi_batch_size = multi_batch_size # multi-lingual batch size
		self.task_batch_size = task_batch_size # task batch size
		self.skip_window = skip_window # skip window for mono-skip gram batch
		self.sen_length = sen_length # upper bar on task input sentence
		self.num_sampled = num_sampled # Number of negative examples to sample.
		self.valid_size = valid_size    # Random set of words to evaluate similarity on.
		self.valid_window = valid_window  # Only pick dev samples in the head of the distribution.
		self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
		self.attention = attention # attention "true"/"false"
		self.attention_size = attention_size
		self.task_tune = task_tune # tune embeddings for task or not "true"/"false"
		self.joint = joint # joint training or not "true"/"false"
		self.num_steps = num_steps # total number of steps
		self.task_mlp_start = task_mlp_start # step to start task 1 : keep low for joint = "true"
		self.logs_path = logs_path # path to log file for tensorboard
		self.num_threads = num_threads # number of threads to use
		self.task_mlp_hidden = task_mlp_hidden # neurons in hidden layer for prediction
		self.skip_gram_learning_rate = skip_gram_learning_rate
		#task_mlp parameters
		self.task_mlp_learning_rate = task_mlp_learning_rate
		self.num_classes = num_classes
		self._init_graph()
		

		print("Class Initialized")

	def _build_dictionaries(self):

		#check if data/processed/mono.p exists
		data_file = Path(DATA_PROCESSED+"mono.p")
		if data_file.is_file():
			print("Loading raw text from previous files")
		else:
			#use this if making processed files
			print("No previous data files found : Creating new files")
			data = read_data(lang_ext=1)
			build_dataset(data,self.vocabulary_size)

		print("Loading Data Files")
		data_mono = cPickle.load(open(DATA_PROCESSED+"mono.p", 'rb'))
		data_bi = cPickle.load(open(DATA_PROCESSED+"bi.p", 'rb'))
		self.dictionary = cPickle.load(open(DATA_PROCESSED+"dictionary.p", 'rb'))
		self.reverse_dictionary = cPickle.load(open(DATA_PROCESSED+"reverse_dictionary.p", 'rb'))
		self.count = cPickle.load(open(DATA_PROCESSED+"count.p", 'rb'))
		return data_mono,data_bi

	def _load_task_mlp_data(self,filename,train_ext='en'):


		task_mlp_file = Path(filename)
		print(task_mlp_file)
		# check if valid file exists, so that we don't reshuffle again
		if task_mlp_file.is_file():
			print("Task data loaded from previous file")
		else:
			print("Creating new task data")
			# creates mlp.p in data/processed folder
			read_dataset_task_mlp(self.dictionary,filename)
		
		#load task dataset
		data_task_mlp = cPickle.load(open(filename, 'rb'))

		# validation data-sets
		data_task_mlp_en_valid = data_task_mlp['en']['dev']
		data_task_mlp_es_valid = data_task_mlp['es']['dev']
		
		# mention training data
		data_task_mlp_train = data_task_mlp['en']['train']

		# create batch for calculating dev/valid accuracy
		# 1. English
		self.task_mlp_en_valid_batch = np.ndarray(shape=(len(data_task_mlp_en_valid),self.sen_length), dtype=np.int32)
		self.task_mlp_en_valid_labels = np.ndarray(shape=(len(data_task_mlp_en_valid),self.num_classes), dtype=np.int32)
		for i in range(len(data_task_mlp_en_valid)):
			self.task_mlp_en_valid_batch[i,:] = pad(data_task_mlp_en_valid[i][0][:self.sen_length], 0, self.sen_length)
			self.task_mlp_en_valid_labels[i,:] = data_task_mlp_en_valid[i][1]

		# 2. Spanish
		self.task_mlp_es_valid_batch = np.ndarray(shape=(len(data_task_mlp_es_valid),self.sen_length), dtype=np.int32)
		self.task_mlp_es_valid_labels = np.ndarray(shape=(len(data_task_mlp_es_valid),self.num_classes), dtype=np.int32)
		for i in range(len(data_task_mlp_es_valid)):
			self.task_mlp_es_valid_batch[i,:] = pad(data_task_mlp_es_valid[i][0][:self.sen_length], 0, self.sen_length)
			self.task_mlp_es_valid_labels[i,:] = data_task_mlp_es_valid[i][1]

#		print(len(data_task_mlp_train))
		return data_task_mlp_train

	def _init_graph(self):

		'''
		Define Graph
		'''

		# initiate graph
		self.graph = tf.Graph()

		with self.graph.as_default(), tf.device('/cpu:0'):
			
			# shared embedding layer
			self.embeddings = tf.Variable(
				tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),
				name='embeddings')
			self.train_skip_inputs = tf.placeholder(tf.int32, name='skip-gram-input')
			self.train_skip_labels = tf.placeholder(tf.int32, name='skip-gram-output')
			
			self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32, name = 'valid-dataset')

			# step to mamnage decay
			self.global_step = tf.Variable(0, trainable=False)

			# Look up embeddings for skip inputs.
			self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_skip_inputs)

			# Construct the variables for the NCE loss
			self.nce_weights = tf.Variable(
				tf.truncated_normal([self.vocabulary_size, self.embedding_size],
					stddev=1.0 / math.sqrt(self.embedding_size)), name='nce_weights')

			self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]), name='nce_biases')

			with tf.name_scope('Skip-gram-NCE-Loss'):
				self.skip_loss = tf.reduce_mean(
					tf.nn.nce_loss(weights=self.nce_weights,
						biases=self.nce_biases,
						labels=self.train_skip_labels,
						inputs=self.embed,
						num_sampled=self.num_sampled,
						num_classes=self.vocabulary_size))

			with tf.name_scope('SGD'):
				self.learning_rate = tf.train.exponential_decay(self.skip_gram_learning_rate, self.global_step,
                                           50000, 0.96, staircase=True)
		#		self.learning_step = (tf.train.GradientDescentOptimizer(self.learning_rate)
		#			.minimize(self.skip_loss, global_step=self.global_step))
				# Construct the SGD optimizer using a learning rate of 1.0.

				self.skip_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.skip_loss,
				 global_step=self.global_step)
		#		self.skip_optimizer = tf.train.GradientDescentOptimizer(self.skip_gram_learning_rate).minimize(self.skip_loss)

			# Create a summary to monitor cost tensor
			tf.summary.scalar("skip_loss", self.skip_loss, collections=['skip-gram'])

			#------------------------ task_mlp Loss and Optimizer ---------------------

			self.train_task_mlp_inputs = tf.placeholder(tf.int32, name='task_mlp-input')
			self.train_task_mlp_labels = tf.placeholder(tf.float32, [None, self.num_classes], name='task_mlp-output')
			self.keep_prob = tf.placeholder("float") 
			self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_task_mlp_inputs)

			
			# attention mechanism used or not
			# if yes, self.attention = 'true'
			# else self.attention = 'false'
			if self.attention == 'true':

				self.embeddings_flat = tf.reshape(self.embed, [-1, self.embedding_size])

				self.embeddings_flat = tf.nn.dropout(self.embeddings_flat, self.keep_prob)
				# zeros insted of random uniform
				self.trans_weights = tf.Variable(tf.zeros([self.embedding_size, self.attention_size]),
				 name='transformation_weights')

#				self.trans_weights = tf.Variable(tf.zeros([self.embedding_size, self.attention_size]),
#					 name='transformation_weights')
				self.trans_bias = tf.Variable(tf.zeros([self.attention_size]), name='trans_bias')

				# task_mlp attention vector
				self.attention_task_mlp = tf.Variable(
					tf.random_uniform([1, self.attention_size], -1.0, 1.0),
					name='attention_vector')

				# Now calculate the attention-weight vector.
				self.keys_flat = tf.tanh(tf.add(tf.matmul(self.embeddings_flat,
					self.trans_weights), self.trans_bias))

				self.keys_flat = tf.nn.dropout(self.keys_flat, self.keep_prob)

				self.keys = tf.reshape(self.keys_flat, tf.concat(0,[tf.shape(self.embed)[:-1], [self.attention_size]]))

#				self.keys = tf.reshape(self.keys_flat, tf.shape(self.embed)[:-1] + [self.attention_size])
#				self.keys = tf.reshape(self.keys_flat, [-1, self.sen_length, self.attention_size])
				self.scores = _attn_mul_fun(self.keys, self.attention_task_mlp)

				self.alignments = nn_ops.softmax(self.scores)

				self.alignments = array_ops.expand_dims(self.alignments,2)

				self.context_vector = math_ops.reduce_sum(self.alignments * 
					self.embed, [1])

				self.context_vector = tf.nn.dropout(self.context_vector, self.keep_prob)

			else:
				self.context_vector = math_ops.reduce_mean(self.embed, [1])



			self.context_vector.set_shape([None, self.embedding_size])


			# Set model weights
			self.W = tf.Variable(tf.zeros([self.embedding_size, self.num_classes]), name='Weights')
			self.b = tf.Variable(tf.zeros([self.num_classes]), name='Bias')
			
			# Construct model and encapsulating all ops into scopes, making
			# Tensorboard's Graph visualization more convenient
			with tf.name_scope('Task-Model'):
    			# Model
				self.pred = tf.nn.softmax(tf.matmul(self.context_vector, self.W) + self.b) # Softmax
			with tf.name_scope('Task-Loss'):
    			# Minimize error using cross entropy
				self.cost = tf.reduce_mean(-tf.reduce_sum(self.train_task_mlp_labels*tf.log(self.pred), reduction_indices=1))
			with tf.name_scope('Task-SGD'):
				self.learning_rate = tf.train.exponential_decay(self.task_mlp_learning_rate, self.global_step,
                                           50000, 0.96, staircase=True)

    			# Gradient Descent
    			self.task_mlp_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost,
				 global_step=self.global_step)
#				self.task_mlp_optimizer = tf.train.GradientDescentOptimizer(self.task_mlp_learning_rate).minimize(self.cost)
			with tf.name_scope('Task-Accuracy'):
    			# Accuracy
				self.acc = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.train_task_mlp_labels, 1))
				self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

			tf.summary.scalar("task_loss", self.cost, collections=['polarity-task'])
			tf.summary.scalar("task_train_accuracy", self.acc, collections=['polarity-task'])

			# Compute the cosine similarity between minibatch examples and all embeddings.
			self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1,
				keep_dims=True))
			self.normalized_embeddings = self.embeddings / self.norm

			self.valid_embeddings = tf.nn.embedding_lookup(
				self.normalized_embeddings, self.valid_dataset)
			self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings,
				transpose_b=True)


			# Add variable initializer.
			self.init_op = tf.initialize_all_variables()
			
			# create a saver
			self.saver = tf.train.Saver()

			self.merged_summary_skip = tf.summary.merge_all('skip-gram')
			self.merged_summary_task_mlp = tf.summary.merge_all('polarity-task')
	#		self.merged_summary_task_mlp_valid = tf.summary.merge_all('task_mlp-valid')


	def fit(self):

		#build dictionaries
		#get mono and parallel data

		data_mono, data_bi = self._build_dictionaries()


		#build validation batches, test data
		#get task_mlp training data
		data_task_mlp_train = self._load_task_mlp_data(DATA_PROCESSED+ "mlp.p")

		print(len(data_task_mlp_train))
		#self._load_task_mlp_data("task_mlp.p")
#		self._init_graph()

		# create a session
		self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.num_threads))

		# with self.sess as session:
		session = self.sess

		session.run(self.init_op)

		average_loss = 0
		task_mlp_average_loss = 0

		# op to write logs to Tensorboard
		summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

		print("Initialized")

		for step in range(self.num_steps):

			if self.joint == 'true':

				if step > self.task_mlp_start:
					task_mlp_batch_inputs, task_mlp_batch_labels = generate_batch_task_mlp(data_task_mlp_train, 
						self.task_batch_size,self.sen_length)

					task_mlp_feed_dict = {self.train_task_mlp_inputs: task_mlp_batch_inputs,
					self.train_task_mlp_labels: task_mlp_batch_labels, self.keep_prob: 0.7}
					
					_, task_mlp_loss_val, summary = session.run([self.task_mlp_optimizer, self.cost,
						self.merged_summary_task_mlp], feed_dict=task_mlp_feed_dict)
					
					summary_writer.add_summary(summary, step)

					task_mlp_average_loss += task_mlp_loss_val
				
				if step % 2 == 0:

					batch_inputs, batch_labels = generate_batch_mono_skip(data_mono, self.batch_size, skip_window=self.skip_window)

				else:

					batch_inputs, batch_labels = generate_batch_multi_skip(data_bi, self.multi_batch_size)

				feed_dict = {self.train_skip_inputs: batch_inputs, self.train_skip_labels: batch_labels}

				_, loss_val,summary = session.run([self.skip_optimizer, self.skip_loss,
					self.merged_summary_skip], feed_dict=feed_dict)

				summary_writer.add_summary(summary, step)

				average_loss += loss_val
			
			if self.joint == 'false':

				if step > self.task_mlp_start:
					task_mlp_batch_inputs, task_mlp_batch_labels = generate_batch_task_mlp(data_task_mlp_train, 
						self.task_batch_size, self.sen_length)
					
					task_mlp_feed_dict = {self.train_task_mlp_inputs: task_mlp_batch_inputs,
					self.train_task_mlp_labels: task_mlp_batch_labels, self.keep_prob: 0.7}
					
					_, task_mlp_loss_val, summary = session.run([self.task_mlp_optimizer, self.cost,
						self.merged_summary_task_mlp], feed_dict=task_mlp_feed_dict)
					
					summary_writer.add_summary(summary, step)

					task_mlp_average_loss += task_mlp_loss_val
				
				else:

					if step % 2 == 0:
						batch_inputs, batch_labels = generate_batch_mono_skip(data_mono, self.batch_size, skip_window=self.skip_window)

					else:
						batch_inputs, batch_labels = generate_batch_multi_skip(data_bi, self.multi_batch_size)

					feed_dict = {self.train_skip_inputs: batch_inputs, self.train_skip_labels: batch_labels}

					_, loss_val,summary = session.run([self.skip_optimizer, self.skip_loss,
						self.merged_summary_skip], feed_dict=feed_dict)

					summary_writer.add_summary(summary, step)

					average_loss += loss_val

			if step % 2000 == 0 and step > self.task_mlp_start:

				# read the validation data batch by batch and compute total accuracy
				total_valid_accuracy = 0

				valid_en_accuracy = self.acc.eval({self.train_task_mlp_inputs:
					self.task_mlp_en_valid_batch, 
					self.train_task_mlp_labels: self.task_mlp_en_valid_labels, self.keep_prob: 1.0}, session=session)
				
				valid_es_accuracy = self.acc.eval({self.train_task_mlp_inputs:
					self.task_mlp_es_valid_batch, 
					self.train_task_mlp_labels: self.task_mlp_es_valid_labels, self.keep_prob: 1.0}, session=session)

				print("Average valid EN acc at ", step, ": ", valid_en_accuracy)
				print("Average valid ES acc at ", step, ": ", valid_es_accuracy)

				summary = tf.Summary(value=[tf.Summary.Value(tag="mlp_valid_EN_accuracy", simple_value=float(valid_en_accuracy))])
				summary_writer.add_summary(summary, step)

				summary = tf.Summary(value=[tf.Summary.Value(tag="mlp_valid_ES_accuracy", simple_value=float(valid_es_accuracy))])
				summary_writer.add_summary(summary, step)

			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
#					task_mlp_average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print("Average loss at step ", step, ": ", average_loss)
				print("Average loss of task_mlp at step ", step, ": ", task_mlp_average_loss)
				average_loss = 0
				task_mlp_average_loss = 0

			
			if step % 10000 == 0:
				sim = self.similarity.eval(session=session)
				for i in xrange(self.valid_size):
					valid_word = self.reverse_dictionary[self.valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = "Nearest to %s:" % valid_word
					for k in xrange(top_k):
						close_word = self.reverse_dictionary[nearest[k]]
						log_str = "%s %s," % (log_str, close_word)
					print(log_str)
			
		final_embeddings = session.run(self.normalized_embeddings)
		self.final_embeddings = final_embeddings
		return self
