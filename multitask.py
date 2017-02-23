#!/usr/bin/python

''' Author : Karan Singla, Dogan Can '''

''' main file for training word embeddings and get sentence embeddings	'''

#from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import Counter
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
import pdb
from pathlib import Path
import commands

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
DATA_MONO_CLEAN = DATA_PROCESSED + "mono_clean/"
commands.getstatusoutput("mkdir -p " + DATA_MONO_CLEAN)

DATA_BI_CLEAN = DATA_PROCESSED + "bi_clean/"
commands.getstatusoutput("mkdir -p " + DATA_BI_CLEAN)

DATA_ID = DATA_PROCESSED + "word2id/"
commands.getstatusoutput("mkdir -p " + DATA_ID)

DATA_BATCH = DATA_PROCESSED + "batch/"
commands.getstatusoutput("mkdir -p " + DATA_BATCH)

#logs_path = './tmp/tensorflow_logs/new/'

# initialize data_indexes
data_mono_index = 0
data_bi_index = 0
#data_task_mlp_index = 0
data_task_mlp_index = {}
data_task_mlp_index['en'] = 0
data_task_mlp_index['es'] = 0
data_task_mlp_index['de'] = 0
data_task_mlp_index['pl'] = 0

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

class DataBuilder(object):
	''' 
	1. this class reads data from monolingual + parallel files
	2. cleans them : read_data(lang_ext=1)
	3. makes dictionary, replace words with integer IDs : 
	build_dataset(bilangs, min_count)
	'''
	def __init__(self,lang_ext=1,min_count=5):

		self.lang_ext = lang_ext
		self.min_count = min_count # minimum count of each word in each language

	def read_data(self):
		"""Extract the first file enclosed in a zip file as a list of words"""

		# cleaning monolingual files and dump clean files
		all_langs = []
		mono_files = glob.glob(DATA_MONO+"*")
		for filename in mono_files:
			print(filename)
			lang = filename.split(".")[-1]
			if lang not in all_langs:
				all_langs.append(lang)
			ext = ":ID:" + lang
			out_file = open(DATA_MONO_CLEAN + os.path.basename(filename) + ".cl",'w')
			with open(filename) as infile:
				for line in infile:
					line = preprocess_text(line)
					if line != '':			
						# lang_ext is sticked to each token
						if self.lang_ext == 1:
							tokens = [x + ext for x in line.split()]
						else:
							tokens = line.split()
						tokens = " ".join(tokens)
						out_file.write(tokens+"\n")
			out_file.close()


		# cleanining bilingual files and dump clean files
		self.bilangs = []
		bi_files = glob.glob(DATA_BI+"*")
		for filename in bi_files:
			print(filename)
			count = 0

			src = filename.split(".")[-1].split("-")[0]
			tgt = filename.split(".")[-1].split("-")[1]
			src_lang = ":ID:" + src
			tgt_lang = ":ID:" + tgt

			if DATA_BI_CLEAN + os.path.basename(filename) not in bilangs:
				bilangs.append(DATA_BI_CLEAN + os.path.basename(filename))

			out_src_file = open(DATA_BI_CLEAN + os.path.basename(filename) + 
				"."+ src + ".cl",'w')
			out_tgt_file = open(DATA_BI_CLEAN + os.path.basename(filename) + 
				"."+ tgt + ".cl",'w')

			with open(filename) as sentence_pair_file:
			
				for sentence_pair_line in sentence_pair_file:
					sentence_pair_line = sentence_pair_line.rstrip()
				
					if len(sentence_pair_line.split(" ||| ")) ==2:
						source_line, target_line = sentence_pair_line.split(" ||| ")

						source_line = preprocess_text(source_line)
						target_line = preprocess_text(target_line)
						count = count + 1
						if source_line != '' and target_line != '':
							source_tokens, target_tokens = source_line.split(' '), target_line.split(' ')
							if self.lang_ext == 1:
								source_tokens = [x + src_lang for x in source_tokens]
								target_tokens = [x + tgt_lang for x in target_tokens]
							source_tokens = ' '.join(source_tokens)
							target_tokens = ' '.join(target_tokens)
							out_src_file.write(source_tokens+"\n")
							out_tgt_file.write(target_tokens+"\n")
			print(count)
			out_src_file.close()
			out_tgt_file.close()


	def build_dataset(self):
		'''
		Build the dictionary and replace rare words with UNK token.
	
		Parameters
		----------
		words: list of tokens
		vocabulary_size: maximum number of top occurring tokens to produce, 
			rare tokens will be replaced by 'UNK'
		'''
		print("Build Dataset and dictionaries")

		# counter for making dictionary	
		wordcount = {}

		# create counter from monolingual data
		mono_files = glob.glob(DATA_MONO_CLEAN+"*")
		for filename in mono_files:
			print(filename)
			lang = os.path.basename(filename).split(".")[-2]
			file = open(filename,'r')
			wordcount[lang] = Counter(file.read().split())
			file.close()
		print("counter created from mono-files")


		# update counter from bilingual
		bi_files = glob.glob(DATA_BI_CLEAN+"*")
		for filename in bi_files:
			print(filename)
			lang = os.path.basename(filename).split(".")[-2]
			file = open(filename,'r')
			wordcount[lang] = wordcount[lang] + Counter(file.read().split())
			file.close()
		print("counter created from bi-files")


		dictionary = dict() # {word : index}
		for lang in wordcount.keys():
			#remove values with freq < self.min_count
			wordcount[lang] = {k:v for k, v in wordcount[lang].items() if v > self.min_count}
			# adding words to dictionaries
			for word in wordcount[lang]:
				dictionary[word] = len(dictionary)
		del wordcount
		print("dictionary created")
		print("Dictionary size",len(dictionary.keys()))


		## replace words by IDs in monolingual data
		data_mono = list()
		for filename in mono_files:
			file = open(filename,'r')
			for line in file:
				line = line.strip().split()
				for i in range(0,len(line)):
					if line[i] in dictionary:
						index = dictionary[line[i]]
					else:
						index = 0
					line[i] = index
				data_mono.append(line)

		random.shuffle(data_mono)
		cPickle.dump(data_mono, open(DATA_ID + 'mono.p', 'wb'))
		del data_mono
		print("mono data created")


		## replace words by IDs in bilingual data
		data_bi = list()
		for filename in self.bilangs:
			print(filename)
			lang1 = os.path.basename(filename).split(".")[1].split("-")[0]
			lang2 = os.path.basename(filename).split(".")[1].split("-")[1]

			lang1_file = open(filename+ "." + lang1 + ".cl").readlines()
			lang2_file = open(filename+ "." + lang2 + ".cl").readlines()

			sent_pair = []
			for i in range(0,len(lang1_file)):
				sent_pair = [lang1_file[i].split(), lang2_file[i].split()]
				pair = []
				for seq in sent_pair:
					for i in range(0,len(seq)):
						if seq[i] in dictionary:
							index = dictionary[seq[i]]
						else:
							index = 0 # dictionary['UNK']
						seq[i] = index
					pair.append(seq)
				data_bi.append(pair)

			del lang1_file
			del lang2_file

		random.shuffle(data_bi)
		cPickle.dump(data_bi, open(DATA_ID + 'bi.p', 'wb'))
		del data_bi # saving memory	
		print("bi data created")

		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

		cPickle.dump(dictionary, open(DATA_ID + 'dictionary.p', 'wb'))
		cPickle.dump(reverse_dictionary, open(DATA_ID + 'reverse_dictionary.p', 'wb'))

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

def generate_batch_data_mono_skip(skip_window=5):

	data_mono = cPickle.load(open(DATA_ID+"mono.p", 'rb'))
	
	batch_mono = open(DATA_BATCH+"mono.csv",'w')

	for sent in data_mono:
		for j in range(skip_window):
			sent = ['<eos>'] + sent + ['<eos>']
		for j in range(skip_window,len(sent)-skip_window):
			for skip in range(1,skip_window+1):
				if sent[j-skip] != '<eos>':
					batch_mono.write(str(sent[j])+","+str(sent[j-skip])+"\n")
				if sent[j+skip] != '<eos>':
					batch_mono.write(str(sent[j])+","+str(sent[j+skip])+"\n")
	batch_mono.close()

def generate_batch_data_multi_skip(window = 5):

	data_bi = cPickle.load(open(DATA_ID+"bi.p", 'rb'))
	
	batch_bi = open(DATA_BATCH+"bi.csv",'w')

	for sent_pair in data_bi:
		sent1 = sent_pair[0]
		sent2 = sent_pair[0]

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
				batch_bi.write(str(sent1[j])+","+str(k)+"\n")

				# l2 -> l1
				batch_bi.write(str(k)+","+str(sent1[j])+"\n")

			for k in sent2[window_low:alignment]:
				# l1 -> l2
				batch_bi.write(str(sent1[j])+","+str(k)+"\n")

				# l2 -> l1
				batch_bi.write(str(k)+","+str(sent1[j])+"\n")


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

	languages = data_task_mlp_train.keys()
	batch = np.ndarray(shape=(task_batch_size*len(languages),sen_length), dtype=np.int32)
	labels = np.ndarray(shape=(task_batch_size*len(languages), 3), dtype=np.int32)
	
	## make a batch which represents all languages

	lang_count = 0

	for key in languages:
		start = lang_count*task_batch_size
		lang_count = lang_count + 1
		for i in range(start,task_batch_size+start):

			sent = data_task_mlp_train[key][data_task_mlp_index[key]]
			sent[0] = pad(sent[0][:sen_length], 0, sen_length)
#			print(sent[0])
			batch[i,:] = sent[0]
			labels[i,:] = sent[1]
			data_task_mlp_index[key] = (data_task_mlp_index[key] + 1) % len(data_task_mlp_train[key])
			if data_task_mlp_index[key] == 0:
				print("shuffle ",key,"task_mlp train data")
				random.shuffle(data_task_mlp_train[key])

	return batch, labels



class MultiTask(BaseEstimator, TransformerMixin):

	def __init__(self, vocabulary_size=500000, embedding_size=200, batch_size=128,
		multi_batch_size=5, task_batch_size=20, skip_window=5, skip_multi_window = 5,
		num_sampled=64, min_count = 10, valid_size=16, valid_window=500, 
		skip_gram_learning_rate=0.02, sen_length=20, task_mlp_learning_rate=0.1,
		num_steps=1400001, task_mlp_start=0, task_mlp_hidden=50, 
		attention='true', attention_size = 150, task_tune='false', joint='true', 
		logs_path='./tmp/tensorflow_logs/test', num_threads=10,num_classes=3,
		train_lang=['en','es','de','pl','ru','sv','bg','pt']):

		# set parameters
		self.vocabulary_size = vocabulary_size # size of vocabulary
		self.embedding_size = embedding_size # Dimension of the embedding vectorself.
		self.batch_size = batch_size # mono-lingual batch size
		self.multi_batch_size = multi_batch_size # multi-lingual batch size
		self.task_batch_size = task_batch_size # task batch size
		self.skip_window = skip_window # skip window for mono-skip gram batch
		self.skip_multi_window = skip_multi_window # window for soft-alignment
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
		self.skip_gram_learning_rate = skip_gram_learning_rate # skip-gram learning rate
		self.min_count = min_count # minimum count of each word
		self.train_lang = train_lang #langids of data to be considered for training
		for lang in self.train_lang:
			data_task_mlp_index[lang] = 0
		#task_mlp parameters
		self.task_mlp_learning_rate = task_mlp_learning_rate
		self.num_classes = num_classes
#		self._init_graph()
		

		print("Class & Graph Initialized")

	def _build_dictionaries(self):

		print("Loading Data Files")
		
		self.dictionary = cPickle.load(open(DATA_ID+"dictionary.p", 'rb'))
		self.reverse_dictionary = cPickle.load(open(DATA_ID+"reverse_dictionary.p", 'rb'))
		print("dictionaries loaded")

		self.vocabulary_size = len(self.dictionary.keys())

	def _load_task_mlp_data(self,filename,train_ext='en', train_lang=['en','es','de','pl']):


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
		data_task_mlp_valid = {}
		for lang in data_task_mlp.keys():	
			data_task_mlp_valid[lang] = data_task_mlp[lang]['dev']

		# mention training data
		data_task_mlp_train = {}
		for lang in train_lang:
				data_task_mlp_train[lang] = data_task_mlp[lang]['train']

		# create batch for calculating dev/valid accuracy
		self.task_mlp_valid_batch = {}
		self.task_mlp_valid_labels = {}

		for lang in data_task_mlp.keys():
			task_mlp_valid_batch = np.ndarray(shape=(len(data_task_mlp_valid[lang]),self.sen_length), dtype=np.int32)
			task_mlp_valid_labels = np.ndarray(shape=(len(data_task_mlp_valid[lang]),self.num_classes), dtype=np.int32)
			for i in range(len(data_task_mlp_valid[lang])):
				task_mlp_valid_batch[i,:] = pad(data_task_mlp_valid[lang][i][0][:self.sen_length], 0, self.sen_length)
				task_mlp_valid_labels[i,:] = data_task_mlp_valid[lang][i][1]

			self.task_mlp_valid_batch[lang] = task_mlp_valid_batch
			self.task_mlp_valid_labels[lang] = task_mlp_valid_labels

#		print(len(data_task_mlp_train))
		return data_task_mlp_train, data_task_mlp.keys()

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
			self.train_skip_inputs, self.train_skip_labels = self.input_pipeline(filenames=[DATA_BATCH+"mono.csv",
					DATA_BATCH+"bi.csv"], batch_size=self.batch_size)
#			self.train_skip_inputs = tf.placeholder(tf.int32, name='skip-gram-input')
#			self.train_skip_labels = tf.placeholder(tf.int32, name='skip-gram-output')
			
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
                                           50000, 0.98, staircase=True)
		

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
                                           50000, 0.98, staircase=True)

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

	def read_my_file_format(self,filename_queue):
		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)
		record_defaults = [[1], [1]]
		col1, col2 = tf.decode_csv(value,record_defaults=record_defaults)
#		example = tf.stack([col1])
		label = tf.stack([col2])
		return col1, label

	def input_pipeline(self,filenames, batch_size, num_epochs=None):
		filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=num_epochs, shuffle=True)
		example, label = self.read_my_file_format(filename_queue)
		# min_after_dequeue defines how big a buffer we will randomly sample
		#   from -- bigger means better shuffling but slower start up and more
		#   memory used.
		# capacity must be larger than min_after_dequeue and the amount larger
		#   determines the maximum we will prefetch.  Recommendation:
		#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
		min_after_dequeue = 10000
		capacity = min_after_dequeue + 3 * batch_size
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
		return example_batch, label_batch

	def fit(self):

		#build dictionaries
		#get mono and parallel data

		self._build_dictionaries()


		#build validation batches, test data
		#get task_mlp training data
		data_task_mlp_train, task_mlp_langs = self._load_task_mlp_data(DATA_ID+ "mlp.p", 
			train_lang=self.train_lang)

		self._init_graph()

#		input_pipeline([DATA_BATCH+"mono.csv",DATA_BATCH+"bi.csv"],self.batch_size)
#		filename_queue = tf.train.string_input_producer([DATA_BATCH+"mono.csv",
#		 DATA_BATCH+"bi.csv"], shuffle=True)

		

		

		
		#self._load_task_mlp_data("task_mlp.p")
#		self._init_graph()

		# create a session
		coord = tf.train.Coordinator()

		self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.num_threads))

		# with self.sess as session:
		session = self.sess

		threads = tf.train.start_queue_runners(sess=session, coord=coord)

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
				
#				batch_inputs, batch_labels = self.input_pipeline(filenames=[DATA_BATCH+"mono.csv",
#					DATA_BATCH+"bi.csv"], batch_size=self.batch_size)
				'''
				if step % 2 == 0:

					batch_inputs, batch_labels = generate_batch_mono_skip(data_mono, self.batch_size, skip_window=self.skip_window)

				else:

					batch_inputs, batch_labels = generate_batch_multi_skip(data_bi, self.multi_batch_size,
					 window=self.skip_multi_window)
				'''
#				feed_dict = {self.train_skip_inputs: batch_inputs, self.train_skip_labels: batch_labels}

				_, loss_val,summary = session.run([self.skip_optimizer, self.skip_loss,
					self.merged_summary_skip])

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
						batch_inputs, batch_labels = generate_batch_multi_skip(data_bi, self.multi_batch_size,
							window=self.skip_multi_window)

					feed_dict = {self.train_skip_inputs: batch_inputs, self.train_skip_labels: batch_labels}

					_, loss_val,summary = session.run([self.skip_optimizer, self.skip_loss,
						self.merged_summary_skip], feed_dict=feed_dict)

					summary_writer.add_summary(summary, step)

					average_loss += loss_val

			if step % 2000 == 0 and step > self.task_mlp_start:

				# read the validation data batch by batch and compute total accuracy
				total_valid_accuracy = 0


				for lang in task_mlp_langs:

					valid_accuracy = self.acc.eval({self.train_task_mlp_inputs:
						self.task_mlp_valid_batch[lang], 
						self.train_task_mlp_labels: self.task_mlp_valid_labels[lang],
						self.keep_prob: 1.0}, session=session)

					print("Average valid "+lang+" acc at ", step, ": ", valid_accuracy)

					summary = tf.Summary(value=[tf.Summary.Value(tag="mlp_valid_"+lang.upper()+"_accuracy",
					 simple_value=float(valid_accuracy))])
					
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
