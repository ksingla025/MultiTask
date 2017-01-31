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

#logs_path = './tmp/tensorflow_logs/new/'

# initialize data_indexes
data_mono_index = 0
data_bi_index = 0
data_task1_index = 0

# Utility Functions

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
			sourcel_line, target_line = sentence_pair_line.split(" ||| ")
			source_tokens, target_tokens = sourcel_line.split(' '), target_line.split(' ')
			if lang_ext == 1:
				source_tokens = [x + src_lang for x in source_tokens]
				target_tokens = [x + tgt_lang for x in target_tokens]
#			source_tokens.extend(['<eos>'])
#			target_tokens.extend(['<eos>'])
			data_bi.append([source_tokens,target_tokens])

	print(len(data_mono),len(data_bi))
	return data_mono,data_bi

def read_dataset_task(dictionary,lang_ext="en",taskID="polarity"):
	task_data = []
	num_classes = 2
	class_counter = 0
	classes = []
	task_files = glob.glob(DATA_TASK+taskID+"/*")
	print(" Reading data for", taskID,"prediction task")
	for file in task_files:
		print(file)
		classes_index = [0]*num_classes
		class_name = ntpath.basename(file).split(".")[-1]
		if class_name not in classes:
			classes.append(class_name)
			classes_index[class_counter] = 1
			class_counter = class_counter + 1

		class_file = open(file,'r')
#		task_data[class_name] = []
		for line in class_file:
			task_entry = []
			line = re.sub(r'[^\w\s]','',line)
			line = line.strip().split()
			line_ID = []
			for word in line:
				word = word+":ID:"+lang_ext
				ID = dictionary.get(word)
				if ID is not None:
					line_ID.append(ID)
			if line_ID != []:
				task_entry.append(line_ID)
				task_entry.append(classes_index)
				task_data.append(task_entry)

	return task_data

def build_dataset(words, vocabulary_size = 200000):
	'''
	Build the dictionary and replace rare words with UNK token.
	
	Parameters
	----------
	words: list of tokens
	vocabulary_size: maximum number of top occurring tokens to produce, 
		rare tokens will be replaced by 'UNK'
	'''
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

	count[0][1] = unk_count # list of tuples (word,count)
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data_mono, data_bi, count, dictionary, reverse_dictionary

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

def generate_batch_multi_skip(data_bi, multi_batch_size):
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
		for j in range(len(sent1)):
			for k in range(len(sent2)):
				batch.append(sent1[j])
				batch.append(sent2[k])
				labels.append(sent2[k])
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

def generate_batch_task1(data_task1_train, task_batch_size, beam_length):
	'''
	generates CBOW batch of the task
	'''
	global data_task1_index
#	beam_length = 20
	batch = []
	labels = []
	batch = np.ndarray(shape=(task_batch_size,beam_length), dtype=np.int32)
	labels = np.ndarray(shape=(task_batch_size, 2), dtype=np.int32)
	for i in range(task_batch_size):
		sent = data_task1_train[data_task1_index]
		sent[0] = pad(sent[0][:beam_length], 0, beam_length)
#		print(sent[0])
		batch[i,:] = sent[0]
		labels[i,:] = sent[1]
		data_task1_index = (data_task1_index + 1) % len(data_task1_train)
#		if data_task1_index == 0:
#			print("shuffle task1 train data")
#			random.shuffle(data_task1_train)

	return batch, labels





class MultiTask(BaseEstimator, TransformerMixin):

	def __init__(self, vocabulary_size=90000, embedding_size=200, batch_size=5,
		multi_batch_size=5, task_batch_size=5, skip_window=2, num_sampled=64,
		valid_size=16, valid_window=500, beam_length=20,task1_learning_rate=0.01,
		num_steps=1400001, task1_start=20000, attention='true',task_tune='false',joint='true',logs_path='./tmp/tensorflow_logs/test', num_threads=10):

		# set parameters
		self.vocabulary_size = vocabulary_size # size of vocabulary
		self.embedding_size = embedding_size # Dimension of the embedding vectorself.
		self.batch_size = batch_size # mono-lingual batch size
		self.multi_batch_size = multi_batch_size # multi-lingual batch size
		self.task_batch_size = task_batch_size # task batch size
		self.skip_window = skip_window # skip window for mono-skip gram batch
		self.beam_length = beam_length # upper bar on task input sentence
		self.num_sampled = num_sampled # Number of negative examples to sample.
		self.valid_size = valid_size    # Random set of words to evaluate similarity on.
		self.valid_window = valid_window  # Only pick dev samples in the head of the distribution.
		self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
		self.attention = attention # attention "true"/"false"
		self.task_tune = task_tune # tune embeddings for task or not "true"/"false"
		self.joint = joint # joint training or not "true"/"false"
		self.num_steps = num_steps # total number of steps
		self.task1_start = task1_start # step to start task 1 : keep low for joint = "true"
		self.logs_path = logs_path # path to log file for tensorboard
		self.num_threads = num_threads # number of threads to use
		#task1 parameters
		self.task1_learning_rate = task1_learning_rate

		self._init_graph()
		

		print("Class Initialized")

	def _build_dictionaries(self):

		data = read_data(lang_ext=1)
		data_mono, data_bi, count, dictionary, reverse_dictionary = build_dataset(data,self.vocabulary_size)
		self.reverse_dictionary = reverse_dictionary
		self.dictionary = dictionary
		self.count = count

		return data_mono,data_bi

	def _load_task1_data(self,filename):

		task1_file = Path(filename)

		# check if valid file exists, so that we don't reshuffle again
		if task1_file.is_file():
			data_task1 = cPickle.load(open(filename, 'rb'))
			self.data_task1_test = data_task1[-530:]
			data_task1_valid = data_task1[-1060:-530]
			data_task1_train = data_task1[:-1060]
			self.task1_valid_batch = np.ndarray(shape=(len(data_task1_valid),self.beam_length), dtype=np.int32)
			self.task1_valid_labels = np.ndarray(shape=(len(data_task1_valid),2), dtype=np.int32)
			for i in range(len(data_task1_valid)):
				self.task1_valid_batch[i,:] = pad(data_task1_valid[i][0][:self.beam_length], 0, self.beam_length)
				self.task1_valid_labels[i,:] = data_task1_valid[i][1]
		else:
			data_task1 = read_dataset_task(self.dictionary,lang_ext="en",taskID="polarity")
			random.shuffle(data_task1)
			self.data_task1_test = data_task1[-530:]
			data_task1_valid = data_task1[-1060:-530]
			data_task1_train = data_task1[:-1060]
			self.task1_valid_batch = np.ndarray(shape=(len(data_task1_valid),self.beam_length), dtype=np.int32)
			self.task1_valid_labels = np.ndarray(shape=(len(data_task1_valid),2), dtype=np.int32)
			for i in range(len(data_task1_valid)):
				self.task1_valid_batch[i,:] = pad(data_task1_valid[i][0][:self.beam_length], 0, self.beam_length)
				self.task1_valid_labels[i,:] = data_task1_valid[i][1]
		return data_task1_train

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
				# Construct the SGD optimizer using a learning rate of 1.0.
				self.skip_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.skip_loss)

			# Create a summary to monitor cost tensor
			tf.summary.scalar("skip_loss", self.skip_loss, collections=['skip-gram'])

			#------------------------ Task1 Loss and Optimizer ---------------------

			self.train_task1_inputs = tf.placeholder(tf.int32, name='task1-input')
			self.train_task1_labels = tf.placeholder(tf.float32, [None, 2], name='task1-output')

			self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_task1_inputs)

			# attention mechanism used or not
			# if yes, self.attention = 'true'
			# else self.attention = 'false'
			if self.attention == 'true':
				# task1 attention vector
				self.attention_task1 = tf.Variable(
					tf.random_uniform([1, self.embedding_size], -1.0, 1.0), name='attention_vector')
			
				self.scores = _attn_mul_fun(self.embed,self.attention_task1)
			
				# Compute alignment weights
				self.alignments = nn_ops.softmax(self.scores)

				# Now calculate the attention-weighted vector.
				self.alignments = array_ops.expand_dims(self.alignments, 2)

				self.context_vector = math_ops.reduce_sum(self.alignments * 
					self.embed, [1])

			else:
				self.context_vector = math_ops.reduce_mean(self.embed, [1])

			self.context_vector.set_shape([None, self.embedding_size])

			# Set model weights
			self.W = tf.Variable(tf.zeros([self.embedding_size, 2]), name='Weights')
			self.b = tf.Variable(tf.zeros([2]), name='Bias')

			# Construct model and encapsulating all ops into scopes, making
			# Tensorboard's Graph visualization more convenient
			with tf.name_scope('Task-Model'):
    			# Model
				self.pred = tf.nn.softmax(tf.matmul(self.context_vector, self.W) + self.b) # Softmax
			with tf.name_scope('Task-Loss'):
    			# Minimize error using cross entropy
				self.cost = tf.reduce_mean(-tf.reduce_sum(self.train_task1_labels*tf.log(self.pred), reduction_indices=1))
			with tf.name_scope('Task-SGD'):
    			# Gradient Descent
				self.task1_optimizer = tf.train.GradientDescentOptimizer(self.task1_learning_rate).minimize(self.cost)
			with tf.name_scope('Task-Accuracy'):
    			# Accuracy
				self.acc = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.train_task1_labels, 1))
				self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

			tf.summary.scalar("task_loss", self.cost, collections=['polarity-task'])
			tf.summary.scalar("task_train_accuracy", self.acc, collections=['polarity-task'])
			tf.summary.scalar("task1_valid_accuracy", self.acc, collections=['task1-valid'])

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
			self.merged_summary_task1 = tf.summary.merge_all('polarity-task')
			self.merged_summary_task1_valid = tf.summary.merge_all('task1-valid')


	def fit(self):

		#build dictionaries
		#get mono and parallel data
		data_mono, data_bi = self._build_dictionaries()

		#build validation batches, test data
		#get task1 training data
		data_task1_train = self._load_task1_data("task1.p")

		#self._load_task1_data("task1.p")
#		self._init_graph()

		# create a session
		self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.num_threads))

		# with self.sess as session:
		session = self.sess

		session.run(self.init_op)

		average_loss = 0
		task1_average_loss = 0

		# op to write logs to Tensorboard
		summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

		print("Initialized")

		for step in range(self.num_steps):

			if self.joint == 'true':

				if step > self.task1_start:
					task1_batch_inputs, task1_batch_labels = generate_batch_task1(data_task1_train, 
						self.task_batch_size,self.beam_length)
					
					task1_feed_dict = {self.train_task1_inputs: task1_batch_inputs,
					self.train_task1_labels: task1_batch_labels}
					
					_, task1_loss_val, summary = session.run([self.task1_optimizer, self.cost,
						self.merged_summary_task1], feed_dict=task1_feed_dict)
					
					summary_writer.add_summary(summary, step)

					task1_average_loss += task1_loss_val
				
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

				if step > self.task1_start:
					task1_batch_inputs, task1_batch_labels = generate_batch_task1(data_task1_train, 
						self.task_batch_size, self.beam_length)
					
					task1_feed_dict = {self.train_task1_inputs: task1_batch_inputs,
					self.train_task1_labels: task1_batch_labels}
					
					_, task1_loss_val, summary = session.run([self.task1_optimizer, self.cost,
						self.merged_summary_task1], feed_dict=task1_feed_dict)
					
					summary_writer.add_summary(summary, step)

					task1_average_loss += task1_loss_val
				
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

			if step % 2000 == 0 and step > self.task1_start:

				_, summary = session.run([self.acc, self.merged_summary_task1_valid],
					feed_dict={self.train_task1_inputs: self.task1_valid_batch, 
					self.train_task1_labels: self.task1_valid_labels})

				valid_accuracy = self.acc.eval({self.train_task1_inputs: self.task1_valid_batch, 
					self.train_task1_labels: self.task1_valid_labels}, session=session)
				tf.summary.scalar("task1_valid_accuracy",valid_accuracy)
				summary_writer.add_summary(summary, step)

			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
#					task1_average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print("Average loss at step ", step, ": ", average_loss)
				print("Average loss of task1 at step ", step, ": ", task1_average_loss)
				average_loss = 0
				task1_average_loss = 0

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

c = MultiTask()
c.fit()
