22#!/usr/bin/python

''' Author : Karan Singla, Dogan Can '''

''' main file for training word embeddings and get sentence embeddings	'''


#standard python imports
import sys
from imp import reload
#reload(sys)
#sys.setdefaultencoding('utf-8')

#standard python imports
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
import _pickle as cPickle
import pdb
from pathlib import Path

#library imports
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from sklearn.base import BaseEstimator, TransformerMixin

# external library imports
#from utils.twokenize import *
from path import *

############### Utility Functions ####################

def preprocess_text(text):
	
	text = text.strip()
	text = text.split()
	text = ' '.join(text)
	text = text.lower()
	return text

def _attn_mul_fun(keys, query):

	return math_ops.reduce_sum(keys * query, [2])

def pad(l, content, width):
	
	l.extend([content] * (width - len(l)))
	return l

def document_pad(document, content, max_sent_len, doc_length, sent_length):

	# pad sentences to sen_len
	document = document[:doc_length]
	sent_length = sent_length[:doc_length]

	for i in range(0,len(document[:doc_length])):
		document[i] = pad(document[i][:max_sent_len], 0, max_sent_len)

	# pad sentences to the document
	if len(document) < doc_length:
		pad_sent = [content] * max_sent_len
		for i in range(0,(doc_length - len(document))):
			document.append(pad_sent)
			sent_length.append(0)

	return document, sent_length

def loss(x1, x2, y, margin = 0.0):
    ''' 
	calucaltes loss depending on cosine similarity and labels
	if label == 1:
		loss = 1 - cosine
	else:
		loss = max(0,cosine - margin)
	x1 : a 2D tensor ( batch_size, embed)
	x2 : a 2D tensor
	y : batch label tensor
	margin : margin for negtive samples loss
    '''

    #take dot product of x1,x2 : [batch_size,1]
    dot_products = tf.reduce_sum(tf.multiply(x1,x2),axis=1)

    # calulcate magnitude of two 1d tensors
    x1_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x1,x1),axis=1))
    x2_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x2,x2),axis=1))

    # calculate cosine distances between them
    cosine = dot_products / tf.multiply(x1_magnitudes,x2_magnitudes)

    # conver it into float and make it a row vector
    labels = tf.to_float(y)
    labels = tf.transpose(labels,[1,0])


    # you can try margin parameters, margin helps to set bound for mismatch cosine
    margin = tf.constant(margin)     

    # calculate number of match and mismatch pairs
    total_labels = tf.to_float(tf.shape(labels)[1])
    match_size = tf.reduce_sum(labels)
    mismatch_size = tf.subtract(total_labels,match_size)

    # loss culation for match and mismatch separately
    match_loss = 1 - cosine
    mismatch_loss = tf.maximum(0., tf.subtract(cosine, margin), 'mismatch_term')

    # combined loss for a batch
    loss_match = tf.reduce_sum(tf.multiply(labels, match_loss))
    loss_mismatch = tf.reduce_sum(tf.multiply((1-labels), mismatch_loss))

    # combined total loss
    # if label is 1, only match_loss will count, otherwise mismatch_loss
    loss = tf.add(tf.multiply(labels, match_loss), \
                  tf.multiply((1 - labels), mismatch_loss), 'loss_add')

    # take average for losses according to size
    loss_match_mean = tf.divide(loss_match,match_size)
    loss_mismatch_mean = tf.divide(loss_mismatch, mismatch_size)
    loss_mean = tf.divide(tf.reduce_sum(loss),total_labels)

    return loss_mean, loss_match_mean, loss_mismatch_mean
#    return loss_mean

def triplet_loss(x1, x2, x3, doc_len, margin = 0.0):
	'''
	x1, x2, x3 is a single document with aligned sentences
	x1, x2 are similar, whereas x3 is different from both
	'''
	# only take actual length of the document
	x1 = x1[:doc_len]
	x2 = x2[:doc_len]
	x3 = x3[:doc_len]

	# calulcate magnitude of two 1d tensors
	x1_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x1,x1),axis=1))
	x2_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x2,x2),axis=1))
	x3_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x3,x3),axis=1))

	x1_magnitudes = tf.add(x1_magnitudes,0.1)
	x2_magnitudes = tf.add(x2_magnitudes,0.1)
	x3_magnitudes = tf.add(x3_magnitudes,0.1)

	#take dot product of x1,x2 : [batch_size,1]
	dot_products_x1x2 = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
	dot_products_x1x3 = tf.reduce_sum(tf.multiply(x1,x3),axis=1)
	dot_products_x2x3 = tf.reduce_sum(tf.multiply(x2,x3),axis=1)

	dot_products_x1x2 = tf.add(dot_products_x1x2,0.0001)
	dot_products_x1x3 = tf.add(dot_products_x1x3,0.0001)
	dot_products_x2x3 = tf.add(dot_products_x1x3,0.0001)

	# calculate cosine distances between them
	cosine_x1x2 = dot_products_x1x2 / tf.multiply(x1_magnitudes,x2_magnitudes)
	cosine_x1x3 = dot_products_x1x3 / tf.multiply(x1_magnitudes,x3_magnitudes)
	cosine_x2x3 = dot_products_x2x3 / tf.multiply(x2_magnitudes,x3_magnitudes)

	print("cosine_x1x2", cosine_x1x2)
	print("cosine_x1x3", cosine_x1x3)
	# you can try margin parameters, margin helps to set bound for mismatch cosine
	margin = tf.constant(margin)

	# loss culation for match and mismatch separately
	match_loss = 1 - cosine_x1x2
	mismatch_loss_x1x3 = tf.maximum(0., tf.subtract(cosine_x1x3, margin), 'mismatch_term_x1x3')
	mismatch_loss_x2x3 = tf.maximum(0., tf.subtract(cosine_x2x3, margin), 'mismatch_term_x2x3')
	mismatch_loss = tf.add(mismatch_loss_x1x3, mismatch_loss_x2x3, 'mismatch_loss_add')

	doc_len = tf.to_float(doc_len)
	# combined loss for a batch
	loss_match = tf.reduce_sum(match_loss)
	loss_mismatch = tf.reduce_sum(mismatch_loss)

	loss = tf.add(loss_match, loss_mismatch)

	return loss, loss_match, loss_mismatch,cosine_x1x2, x1_magnitudes, x2_magnitudes, x3_magnitudes 


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

	data_bi = cPickle.load(open(DATA_ID+"bi_train.p", 'rb'))
	
	batch_bi = open(DATA_BATCH+"bi_train.csv",'w')

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

#### ---- helper function for creating batch data ---- ####

def random_document_picker(ted_corpus,mode='train'):
	random_lang1 = random.choice(list(ted_corpus.keys()))
	random_lang2 = random.choice(list(ted_corpus[random_lang1].keys()))

	corpus = ted_corpus[random_lang1][random_lang2][mode]

	random_key = random.choice(list(corpus.keys()))
	random_key2 = random.choice(list(corpus[random_key].keys()))

	random_filename = random.choice(list(corpus[random_key][random_key2].keys()))

	return corpus[random_key][random_key2][random_filename]

def random_sentence_picker(ted_corpus,mode='train'):
	random_lang1 = random.choice(list(ted_corpus.keys()))
	random_lang2 = random.choice(list(ted_corpus[random_lang1].keys()))

	corpus = ted_corpus[random_lang1][random_lang2][mode]

	random_key = random.choice(list(corpus.keys()))
	random_key2 = random.choice(list(corpus[random_key].keys()))

	random_filename = random.choice(list(corpus[random_key][random_key2].keys()))

	while len(corpus[random_key][random_key2][random_filename]) == 0:
		random_filename = random.choice(list(corpus[random_key][random_key2].keys()))

	random_sent = random.choice(corpus[random_key][random_key2][random_filename])
	
	return random_sent

#--------------------------------------------------------------#

def generate_batch_data_task_docsim(langpair=['en-de'], max_sent_len = 32, max_doc_size = 30):

	ted_corpus = cPickle.load(open(DATA_ID+"ted.p", 'rb'))

	print("ted corpus loaded")

	lang1 = langpair[0].split('-')[0]
	lang2 = langpair[0].split('-')[1]

	# get lang1 train corpus
	lang1_train_corpus = ted_corpus[lang1][lang2]['train']
	lang1_train_corpus_keys = list(lang1_train_corpus.keys())
	random.shuffle(lang1_train_corpus_keys)

	print(lang1_train_corpus_keys)
	print("Creating Epoch data for TED document similarity")
	# get lang2 train corpus
	lang2_train_corpus = ted_corpus[lang2][lang1]['train']

	# randomly pick random category
	epoch = []
	sample_count = 0
	for key in lang1_train_corpus_keys:

		lang1_train_corpus_key_keys = list(lang1_train_corpus[key].keys())
		random.shuffle(lang1_train_corpus_key_keys)

		# randomly pick positive / negative
		for key2 in lang1_train_corpus_key_keys:

			for filename in lang1_train_corpus[key][key2]:

				if filename in lang2_train_corpus[key][key2].keys():

					sample_count = sample_count + 1
					document_len = []
					sequence_length = []

					#document 1 
					doc1 = lang1_train_corpus[key][key2][filename]
					if len(doc1) > max_doc_size:
						doc1len = max_doc_size
						doc3len = doc1len
					else:
						doc1len = len(doc1)
						doc3len = doc1len
						assert doc1len != 0
					document_len.append(doc1len)

					sent_length = []
					for line in doc1:
						sent_length.append(len(line))
					doc1, sent_length = document_pad(doc1, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					sequence_length.append(sent_length)

					#document 2
					doc2 = lang2_train_corpus[key][key2][filename]
					if len(doc2) > max_doc_size:
						doc2len = max_doc_size
					else:
						doc2len = len(doc2)
					document_len.append(doc2len)

					sent_length = []
					for line in doc2:
						sent_length.append(len(line))
					doc2, sent_length = document_pad(doc2, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					sequence_length.append(sent_length)

					#document 3 : negative sentences
					doc3 = []
					sent_length = []
					for i in range(0,doc3len):
						random_sent = random_sentence_picker(ted_corpus)
						doc3.append(random_sent)
						sent_length.append(len(random_sent))

					if doc3len > max_doc_size:
						doc3len = max_doc_size
					document_len.append(doc3len)

					doc3, sent_length = document_pad(doc3, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					sequence_length.append(sent_length)

					#document 4 : negative document
					doc4 = random_document_picker(ted_corpus,mode='train')
					if len(doc4) > max_doc_size:
						doc4len = max_doc_size
					else:
						doc4len = len(doc4)
					document_len.append(doc4len)

					sent_length = []
					for line in doc4:
						sent_length.append(len(line))
					doc4, sent_length = document_pad(doc4, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					assert len(doc4) != 0
					sequence_length.append(sent_length)


					sample = [doc1] + [doc2] + [doc3] + [doc4] + [document_len] + [sequence_length]
					epoch.append(sample)
	print("Data Created : total samples",sample_count)
	return epoch

def generate_batch_data_task_sentsim(max_length=32, neg_sample = 1):
	''' 
	generate batch for sentence similarity
	'''

	data_bi = cPickle.load(open(DATA_ID+"bi_train.p", 'rb'))
	data_mono = cPickle.load(open(DATA_ID+"mono.p", 'rb'))

	# batch file for training sentence similarity
	batch_sentsim = open(DATA_BATCH+"sentsim.csv",'w')
	for pair in data_bi:

		# make length of each sentence to max length
		pair[0] = pad(pair[0][:max_length], 0, max_length)
		pair[1] = pad(pair[1][:max_length], 0, max_length)
		sent1 = ",".join(str(x) for x in pair[0])
		sent2 = ",".join(str(x) for x in pair[1])

		batch_sentsim.write(sent1+","+sent2+",1\n")

		for i in range(0,neg_sample):

			# we add a random sentence from monolingual sentence to say, that it's not similar to it
			rand_mono = random.choice(data_mono)
			rand_mono = pad(rand_mono[:max_length], 0, max_length)
			negative = ",".join(str(x) for x in rand_mono)

			batch_sentsim.write(sent1+","+negative+",0\n")
			batch_sentsim.write(sent2+","+negative+",0\n")

	batch_sentsim.close()

	del data_bi #saving memory

	data_valid = cPickle.load(open(DATA_ID+"bi_valid.p", 'rb'))

	print(data_valid.keys())
	for key in data_valid.keys():
		#filename of the valid file
		filename = "valid_"+key.replace(":","_")+".csv"

		batch_valid = open(DATA_BATCH+filename,'w')

		for pair in data_valid[key]:

			pair[0] = pad(pair[0][:max_length], 0, max_length)
			pair[1] = pad(pair[1][:max_length], 0, max_length)
			sent1 = ",".join(str(x) for x in pair[0])
			sent2 = ",".join(str(x) for x in pair[1])

			batch_valid.write(sent1+","+sent2+",1\n")

			# we add a random sentence from monolingual sentence to say, that it's not similar to it
			rand_mono = random.choice(data_mono)
			rand_mono = pad(rand_mono[:max_length], 0, max_length)
			negative = ",".join(str(x) for x in rand_mono)

			batch_valid.write(sent1+","+negative+",0\n")
			batch_valid.write(sent2+","+negative+",0\n")

		batch_valid.close()


def BiRNN(lstm_bw_cell, x, sequence_length, seq_max_len=32,idd='sent'):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps

	# Get lstm cell output
	with tf.variable_scope(idd+'lstm1', reuse=True):
		outputs, states = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32, sequence_length=sequence_length)

	return outputs

class Aggregator(object):


	def __init__(self,x, sequence_length,embedding_size, attention_size, n_hidden=100, lstm_layer=1, attention=1, keep_prob=0.7,idd='sent'):
		
		self.idd = idd
		self.trans_bias = tf.Variable(tf.zeros([attention_size]), name=self.idd+'_trans_bias')
		
		self.attention_task = tf.Variable(tf.random_uniform([1, attention_size], -1.0, 1.0),
			name=self.idd+'attention_vector')

		self.embedding_size = embedding_size
		self.attention_size = attention_size
		self.n_hidden = n_hidden # hidden layer num of features
		self.keep_prob = keep_prob
		self.attention = attention
		self.sequence_length = sequence_length


		if lstm_layer == 1:
			# Define lstm cells with tensorflow
			# Forward direction cell
			self.trans_weights = tf.Variable(tf.zeros([self.n_hidden, attention_size]),
			name=self.idd+'transformation_weights')

			with tf.variable_scope(self.idd+'backward'):
				self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

			with tf.variable_scope(self.idd+'lstm1'):
				outputs, states = tf.nn.dynamic_rnn(self.lstm_bw_cell, x, dtype=tf.float32,sequence_length=sequence_length)
			# Backward direction cell
#			with tf.variable_scope('backward'):
#				self.lstm_fw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

		else:
			self.trans_weights = tf.Variable(tf.zeros([embedding_size, attention_size]),
			name=self.idd+'transformation_weights')



	def attention_based_aggregator(self, embed):

		# make the embeddings flat [batch_size*sen_length*embedding_size,1]

		if self.attention == 0:

			context_vector = math_ops.reduce_mean(embed, [1])

		else:

			embeddings_flat = tf.reshape(embed, [-1, self.embedding_size])

			# Now calculate the attention-weight vector.

			# tanh transformation of embeddings
			keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
				self.trans_weights), self.trans_bias))

			# reshape the keys according to our embed vector
			keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))

			# calculate score for each word embedding and take softmax on it
			scores = math_ops.reduce_sum(keys * self.attention_task, [2])
			alignments = nn_ops.softmax(scores)

			# expand aligments dimension so that we can multiply it with embed tensor
			alignments = array_ops.expand_dims(alignments,2)

			# generate context vector by making 
			context_vector = math_ops.reduce_sum(alignments * 
				embed, [1])

		return context_vector

	def attention_based_aggregator_with_lstm(self, embed, sequence_length):


		# get BiRNN outputs
		outputs = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)
		outputs = tf.nn.dropout(outputs, self.keep_prob)

		if self.attention == 0:

			context_vector = math_ops.reduce_mean(outputs, [1])

		else:

			embeddings_flat = tf.reshape(outputs, [-1, self.n_hidden])


			# tanh transformation of embeddings
			keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
				self.trans_weights), self.trans_bias))

			# reshape the keys according to our embed vector
			keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(outputs)[:-1], [self.attention_size]]))
		
			# calculate score for each word embedding and take softmax on it
			scores = _attn_mul_fun(keys, self.attention_task)
			alignments = nn_ops.softmax(scores)

			# expand aligments dimension so that we can multiply it with embed tensor
			alignments = array_ops.expand_dims(alignments,2)

			# generate context vector by making 
			context_vector = math_ops.reduce_sum(alignments * 
				outputs, [1])

		return context_vector


class MultiTask(BaseEstimator, TransformerMixin):

	def __init__(self, embedding_size=200, batch_size=256,
		multi_batch_size=5, docsim_batch_size=5, skip_window=5, skip_multi_window = 5,
		num_sampled=64, min_count = 5, valid_size=16, valid_window=500, 
		skip_gram_learning_rate=0.01, sen_length=20, sentsim_learning_rate=0.0005,
		num_steps=1400001, task_mlp_start=0, task_mlp_hidden=50, 
		attention=1, n_hidden=100, attention_size = 150, joint='true', 
		logs_path= 'test', max_length=32, lstm_layer=1, keep_prob = 0.7,
		num_threads=10,num_classes=2, loss_margin=0.0):

		# set parameters

		self.embedding_size = embedding_size # Dimension of the embedding vectorself.
		self.batch_size = batch_size # mono-lingual batch size
		self.multi_batch_size = multi_batch_size # multi-lingual batch size
		self.docsim_batch_size = docsim_batch_size
		self.skip_window = skip_window # skip window for mono-skip gram batch
		self.skip_multi_window = skip_multi_window # window for soft-alignment
		self.sen_length = sen_length # upper bar on task input sentence
		self.num_sampled = num_sampled # Number of negative examples to sample.
		self.valid_size = valid_size    # Random set of words to evaluate similarity on.
		self.valid_window = valid_window  # Only pick dev samples in the head of the distribution.
		self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
		self.attention = attention # attention 1/0
		self.lstm_layer = lstm_layer # method lstm layer or not 1/0
		self.attention_size = attention_size
		self.joint = joint # joint training or not "true"/"false"
		self.num_steps = num_steps # total number of steps
		self.task_mlp_start = task_mlp_start # step to start task 1 : keep low for joint = "true"
		self.logs_path = LOGS_PATH + logs_path # path to log file for tensorboard
		self.num_threads = num_threads # number of threads to use
		self.task_mlp_hidden = task_mlp_hidden # neurons in hidden layer for prediction
		self.skip_gram_learning_rate = skip_gram_learning_rate # skip-gram learning rate
		self.min_count = min_count # minimum count of each word
		#task_mlp parameters
		self.sentsim_learning_rate = sentsim_learning_rate
		self.num_classes = num_classes
		self.n_hidden = n_hidden # hiddent units for LSTM cell
		self.max_length = max_length
		self.loss_margin = loss_margin

		self.docsim_data_index = 0
		self.docsim_data = generate_batch_data_task_docsim(langpair=['en-de'], max_sent_len = 100, max_doc_size = 300)
		
		# initiate graph
		self.graph = tf.Graph()
		self._build_dictionaries()
		self._init_graph()

		print("Class & Graph Initialized")

	def _build_dictionaries(self):

		print("Loading Data Files")
		
		self.dictionary = cPickle.load(open(DATA_ID+"dictionary.p", 'rb'))
		self.reverse_dictionary = cPickle.load(open(DATA_ID+"reverse_dictionary.p", 'rb'))
		print("dictionaries loaded")

		self.vocabulary_size = len(self.dictionary.keys())

	def _generate_batch_docsim(self):

		doc1_batch = []
		doc2_batch = []
		doc3_batch = []
		doc4_batch = []
		seq_len_batch = []
		doc_len_batch = []

		for i in range(0,self.docsim_batch_size):
			doc1, doc2, doc3, doc4, doc_len, seq_len = self.docsim_data[self.docsim_data_index]

			assert len(seq_len) == 4

			doc1_batch.append(doc1)
			doc2_batch.append(doc2)
			doc3_batch.append(doc3)
			doc4_batch.append(doc4)
			seq_len_batch.append(seq_len)
			doc_len_batch.append(doc_len)

			self.docsim_data_index = (self.docsim_data_index + 1) % len(self.docsim_data)

			if self.docsim_data_index == 0:
				print("generating new epoch for document similarity")
				self.docsim_data = generate_batch_data_task_docsim(langpair=['en-de'], max_sent_len = 100, max_doc_size = 300)


		return np.array(doc1_batch), np.array(doc2_batch), np.array(doc3_batch), np.array(doc4_batch), np.array(seq_len_batch), np.array(doc_len_batch)




	def docsim_task_graph(self):

		# input document batches, shape : batch_size*docsize
		self.doc1 = tf.placeholder(tf.int32, [self.docsim_batch_size,300,100], name='doc1')
		self.doc2 = tf.placeholder(tf.int32, [self.docsim_batch_size,300,100], name='doc2')
		self.doc3 = tf.placeholder(tf.int32, [self.docsim_batch_size,300,100], name='doc3')
		self.doc4 = tf.placeholder(tf.int32, [self.docsim_batch_size,300,100], name='doc4')

		# unstack each document batch to list of documents
		self.doc1unstack = tf.unstack(self.doc1)
		self.doc2unstack = tf.unstack(self.doc2)
		self.doc3unstack = tf.unstack(self.doc3)
		self.doc4unstack = tf.unstack(self.doc4)

		# document lengths of each sentence in each batch for doc1,doc2,doc3,doc4
		self.seq_len = tf.placeholder(tf.int32, [self.docsim_batch_size,4,300], name='seq-len')
		self.seq_len_doc1 = tf.slice(self.seq_len, [0, 0, 0], [self.docsim_batch_size, 1, 300])
		self.seq_len_doc2 = tf.slice(self.seq_len, [0, 1, 0], [self.docsim_batch_size, 1, 300])
		self.seq_len_doc3 = tf.slice(self.seq_len, [0, 2, 0], [self.docsim_batch_size, 1, 300])
		self.seq_len_doc4 = tf.slice(self.seq_len, [0, 3, 0], [self.docsim_batch_size, 1, 300])

		# unstack each document seq ken to a list of batch size
		self.seq_len_doc1unstack = tf.unstack(self.seq_len_doc1)
		self.seq_len_doc2unstack = tf.unstack(self.seq_len_doc2)
		self.seq_len_doc3unstack = tf.unstack(self.seq_len_doc3)
		self.seq_len_doc4unstack = tf.unstack(self.seq_len_doc4)


		self.doc_len = tf.placeholder(tf.int32, [self.docsim_batch_size,4], name='doc-len')
		self.doc_len_doc1 =  tf.unstack(tf.reshape(tf.slice(self.doc_len, [0, 0], [self.docsim_batch_size, 1]), [-1]))
		self.doc_len_doc2 =  tf.unstack(tf.reshape(tf.slice(self.doc_len, [0, 1], [self.docsim_batch_size, 1]), [-1]))
		self.doc_len_doc3 =  tf.unstack(tf.reshape(tf.slice(self.doc_len, [0, 2], [self.docsim_batch_size, 1]), [-1]))
		self.doc_len_doc4 =  tf.unstack(tf.reshape(tf.slice(self.doc_len, [0, 3], [self.docsim_batch_size, 1]), [-1]))

		self.keep_prob = tf.placeholder("float")

		# initialize aggregator		
		with tf.name_scope('Sent_AttentionBasedAggregator'):

			self.embed = tf.nn.embedding_lookup(self.embeddings, self.doc1unstack[0])
			self.sequence_length = tf.reshape(self.seq_len_doc1unstack[0],[-1])
			print("sent1context",tf.shape(self.embed))
			self.sent_attention_aggrgator = Aggregator(x=self.embed, embedding_size=self.embedding_size,
				sequence_length=self.sequence_length, attention_size=self.attention_size, n_hidden=self.n_hidden, lstm_layer=self.lstm_layer,
				attention=self.attention, keep_prob=self.keep_prob, idd='sent')
		
		self.doc1context = []
		self.doc2context = []
		self.doc3context = []
		self.doc4context = []

		self.totalsent_loss = []
		self.totalsent_loss_match = []
		self.totalsent_loss_mismatch = []
		
		for i in range(0,len(self.doc1unstack)):

			'''
			1. get embeddings for each sentence
			2. get sequence length of each sentence and convert a flat vector of document size
			3. generate embeddings for each sentence
			'''
			self.embed = tf.nn.embedding_lookup(self.embeddings, self.doc1unstack[i])
			self.sequence_length = tf.reshape(self.seq_len_doc1unstack[i],[-1])
			self.doc1context_single = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.embed, 
				self.sequence_length)
			self.doc1context.append(self.doc1context_single)
		
			'''
			1. get embeddings for each sentence
			2. get sequence length of each sentence and convert a flat vector of document size
			3. generate embeddings for each sentence
			'''
			self.embed = tf.nn.embedding_lookup(self.embeddings, self.doc2unstack[i])
			self.sequence_length = tf.reshape(self.seq_len_doc2unstack[i],[-1])
			self.doc2context_single = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.embed, 
				self.sequence_length)
			self.doc2context.append(self.doc2context_single)
		
			'''
			1. get embeddings for each sentence
			2. get sequence length of each sentence and convert a flat vector of document size
			3. generate embeddings for each sentence
			'''
			self.embed = tf.nn.embedding_lookup(self.embeddings, self.doc3unstack[i])
			self.sequence_length = tf.reshape(self.seq_len_doc3unstack[i],[-1])
			self.doc3context_single = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.embed, 
				self.sequence_length)
			self.doc3context.append(self.doc3context_single)

			# get total sentence level loss for sentence aligned document
			with tf.name_scope('doc-sent-Loss'):
				self.loss_single = triplet_loss(self.doc1context_single, self.doc2context_single, self.doc3context_single,
					doc_len=self.doc_len_doc1[i], margin=self.loss_margin)

			'''
			loss_single
			0 : loss
			1 : loss_match
			2 : loss_mismatch
			'''
			self.totalsent_loss.append(self.loss_single[0])
			self.totalsent_loss_match.append(self.loss_single[1])
			self.totalsent_loss_mismatch.append(self.loss_single[2])

		# doc4 can have any length which is not equal to doc1, doc2, doc3
		for i in range(0,len(self.doc4unstack)):

			'''
			1. get embeddings for each sentence
			2. get sequence length of each sentence and convert a flat vector of document size
			3. generate embeddings for each sentence
			'''

			self.embed = tf.nn.embedding_lookup(self.embeddings, self.doc4unstack[i])
			self.sequence_length = tf.reshape(self.seq_len_doc4unstack[i],[-1])
			self.doc4context_single = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.embed, 
				self.sequence_length)
			self.doc4context.append(self.doc4context_single)
	
		
		'''
		1. stack losses of all sentences
		2. take an average of them
		'''
		self.totalsent_loss = tf.stack(self.totalsent_loss)
		self.totalsent_loss = tf.reduce_sum(self.totalsent_loss)

		self.totalsent_loss_match = tf.stack(self.totalsent_loss_match)
		self.totalsent_loss_match = tf.reduce_sum(self.totalsent_loss_match)

		self.totalsent_loss_mismatch = tf.stack(self.totalsent_loss_mismatch)
		self.totalsent_loss_mismatch = tf.reduce_sum(self.totalsent_loss_mismatch)


		#stack all (doc_batch_size) doc1 context vectors
		self.doc1totalcontext = tf.stack(self.doc1context)
		self.doc2totalcontext = tf.stack(self.doc2context)
		self.doc3totalcontext = tf.stack(self.doc3context)
		self.doc4totalcontext = tf.stack(self.doc4context)


		############################### Document level analysis ################################

		# doc1context are fo the form batchsize * 300 * n_hidden ( of sentence encoder )
		
		with tf.name_scope('Doc_AttentionBasedAggregator'):

			print("doc1context",tf.shape(self.doc1totalcontext))
			self.doc_attention_aggrgator = Aggregator(x=self.doc1totalcontext, embedding_size=self.embedding_size,
				sequence_length=self.doc_len_doc1, attention_size=self.attention_size, n_hidden=self.n_hidden, lstm_layer=self.lstm_layer,
				attention=self.attention, keep_prob=self.keep_prob, idd='doc')

		self.doc1totalcontext = self.doc_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc1totalcontext, 
				self.doc_len_doc1)
		self.doc2totalcontext = self.doc_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc2totalcontext, 
				self.doc_len_doc2)
		self.doc3totalcontext = self.doc_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc3totalcontext, 
				self.doc_len_doc3)
		self.doc4totalcontext = self.doc_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc4totalcontext, 
				self.doc_len_doc4)

		# take document[1,2,4] and find total contrasitive loss
		self.document_loss = triplet_loss(self.doc1totalcontext, self.doc2totalcontext, self.doc4totalcontext,
				doc_len=self.docsim_batch_size, margin=self.loss_margin)

		loss = tf.add(self.document_loss[0],self.totalsent_loss)

		with tf.name_scope('Task-SGD'):
			self.learning_rate = tf.train.exponential_decay(self.sentsim_learning_rate, self.global_step,
				50000, 0.98, staircase=True)


			self.task_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,
				global_step=self.global_step)


		self.sentenctrans_sentloss = tf.gradients(self.totalsent_loss,[self.sent_attention_aggrgator.trans_weights])
		self.sentenctrans_docloss = tf.gradients(self.document_loss[0],[self.sent_attention_aggrgator.trans_weights])

		tf.summary.histogram("sentencoder_transweight_sentloss", self.sentenctrans_sentloss, collections=['doc2vec-task'])
		tf.summary.histogram("sentencoder_transweight_docloss", self.sentenctrans_docloss, collections=['doc2vec-task'])



		tf.summary.scalar("total_loss", loss, collections=['doc2vec-task'])

		tf.summary.scalar("document_level_loss", self.document_loss[0], collections=['doc2vec-task'])
		tf.summary.scalar("document_level_loss_match", self.document_loss[1], collections=['doc2vec-task'])
		tf.summary.scalar("document_level_loss_mismatch", self.document_loss[2], collections=['doc2vec-task'])

		tf.summary.scalar("sent_loss", self.totalsent_loss, collections=['doc2vec-task'])
		tf.summary.scalar("sent_loss_match", self.totalsent_loss_match, collections=['doc2vec-task'])
		tf.summary.scalar("sent_loss_mismatch", self.totalsent_loss_mismatch, collections=['doc2vec-task'])

#		tf.summary.scalar("task_loss_match_divide", self.cost_match_mean, collections=['polarity-task'])
#		tf.summary.scalar("task_loss_mismatch_divide", self.cost_mismatch_mean, collections=['polarity-task'])

#		self.doc1context = tf.stack(self.doc1context)


		'''
		if self.lstm_layer == 1:

			self.doc1context[i] = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc1embed[i])
			self.doc2context[i] = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc2embed[i])
			self.doc3context[i] = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc3embed[i])
			self.doc4context[i] = self.sent_attention_aggrgator.attention_based_aggregator_with_lstm(self.doc4embed[i])
		'''

	def sentsim_task_graph(self):

		# training batch extractor
		self.train_sentsimx_batch, self.train_sentsimy_batch, self.train_sentsim_labels_batch = self.input_pipeline_sentsim(filenames=[DATA_BATCH+'sentsim.csv'],
			batch_size=self.task_batch_size)

		# validation batch extractor
		self.valid_sentsimx_batch, self.valid_sentsimy_batch, self.valid_sentsim_labels_batch = self.input_pipeline_sentsim(filenames=[DATA_BATCH+'valid_bg_en.csv'],
			batch_size=self.task_batch_size)

		self.train_sentsimx = tf.placeholder(tf.int32, [None,None], name='sentsim-inputx')
		self.train_sentsimy = tf.placeholder(tf.int32, [None,None], name='sentsim-inputy')
		self.train_sentsim_labels = tf.placeholder(tf.float32, [None, 1], name='sentsim-outlabel')

		self.keep_prob = tf.placeholder("float")

		#get embeddings for x and y input sentence
		self.embedx = tf.nn.embedding_lookup(self.embeddings, self.train_sentsimx)
		self.embedx = tf.nn.dropout(self.embedx, self.keep_prob)

		self.embedy = tf.nn.embedding_lookup(self.embeddings, self.train_sentsimy)
		self.embedy = tf.nn.dropout(self.embedy, self.keep_prob)

		# initialize attention aggregator
		with tf.name_scope('Sent_AttentionBasedAggregator'):

			self.sent_attention_aggrgator = Aggregator(x=self.embedx, embedding_size=self.embedding_size,
				attention_size=self.attention_size, n_hidden=self.n_hidden, lstm_layer=self.lstm_layer,
				attention=self.attention, keep_prob=self.keep_prob, scope='sent')

		with tf.name_scope('Doc_AttentionBasedAggregator'):

			self.doc_attention_aggrgator = Aggregator(x=self.embedx, embedding_size=self.embedding_size,
				attention_size=self.attention_size, n_hidden=self.n_hidden, lstm_layer=self.lstm_layer,
				attention=self.attention, keep_prob=self.keep_prob, scope='doc')

		# if using lstm layer
		if self.lstm_layer == 1:
			
			self.contextx = self.attention_aggrgator.attention_based_aggregator_with_lstm(self.embedx)
			self.contextx = tf.nn.dropout(self.contextx, self.keep_prob)

			self.contexty = self.attention_aggrgator.attention_based_aggregator_with_lstm(self.embedy)
			self.contexty = tf.nn.dropout(self.contexty, self.keep_prob)

		# if no lstm layer
		if self.lstm_layer == 0:

			self.contextx = self.attention_aggrgator.attention_based_aggregator(self.embedx)
			self.contextx = tf.nn.dropout(self.contextx, self.keep_prob)

			self.contexty = self.attention_aggrgator.attention_based_aggregator(self.embedy)
			self.contexty = tf.nn.dropout(self.contexty, self.keep_prob)


		with tf.name_scope('Task-Loss'):
			self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean = loss(self.contextx, self.contexty, self.train_sentsim_labels)
#			self.cost = loss(self.contextx, self.contexty, self.train_sentsim_labels,self.loss_margin)
    		# Minimize error using cross entropy
#			self.cost = tf.reduce_mean(-tf.reduce_sum(self.train_sentsim_labels*tf.log(self.pred), axis=1))
		with tf.name_scope('Task-SGD'):
			self.learning_rate = tf.train.exponential_decay(self.sentsim_learning_rate, self.global_step,
				50000, 0.98, staircase=True)


			self.task_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_mean,
				global_step=self.global_step)

		tf.summary.scalar("task_loss_divide", self.cost_mean, collections=['polarity-task'])
		tf.summary.scalar("task_loss_match_divide", self.cost_match_mean, collections=['polarity-task'])
		tf.summary.scalar("task_loss_mismatch_divide", self.cost_mismatch_mean, collections=['polarity-task'])


	def _init_graph(self):

		'''
		Define Graph
		'''

		with self.graph.as_default(), tf.device('/cpu:0'):
			
			# shared embedding layer
			self.embeddings = tf.Variable(
				tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),
				name='embeddings')

			#training batch extractor
			self.train_skip_inputs, self.train_skip_labels = self.input_pipeline(filenames=[DATA_BATCH+"mono.csv",
					DATA_BATCH+"bi_train.csv"], batch_size=self.batch_size)

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

			with tf.name_scope('Skip-gram-SGD'):
				self.learning_rate = tf.train.exponential_decay(self.skip_gram_learning_rate, self.global_step,
                                           50000, 0.98, staircase=True)
		

				self.skip_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.skip_loss,
				 global_step=self.global_step)
		#		self.skip_optimizer = tf.train.GradientDescentOptimizer(self.skip_gram_learning_rate).minimize(self.skip_loss)

			# Create a summary to monitor cost tensor
			tf.summary.scalar("skip_loss", self.skip_loss, collections=['skip-gram'])

			#------------------------ task_mlp Loss and Optimizer ---------------------
			with tf.name_scope('docsim-graph'):
				self.docsim_task_graph()


			# Compute the cosine similarity between minibatch examples and all embeddings.
			self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1,
				keep_dims=True))
			self.normalized_embeddings = self.embeddings / self.norm

			self.valid_embeddings = tf.nn.embedding_lookup(
				self.normalized_embeddings, self.valid_dataset)
			self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings,
				transpose_b=True)


			# Add variable initializer.
			self.init_op = tf.global_variables_initializer()
			
			# create a saver
			self.saver = tf.train.Saver()

			self.merged_summary_skip = tf.summary.merge_all('skip-gram')
			self.merged_summary_task_mlp = tf.summary.merge_all('doc2vec-task')
	#		self.merged_summary_task_mlp_valid = tf.summary.merge_all('task_mlp-valid')

	def read_format_skipgram(self,filename_queue):

		# file reader and value generator
		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)

		# default format of the file
		record_defaults = [[1], [1]]

		# get values in csv files
		col1, col2 = tf.decode_csv(value,record_defaults=record_defaults)
#		
		# col1 is input, col2 is predicted label
		label = tf.stack([col2])
		return col1, label

	def input_pipeline(self,filenames, batch_size, num_epochs=None):
		filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=num_epochs, shuffle=True)
		example, label = self.read_format_skipgram(filename_queue)
		# min_after_dequeue defines how big a buffer we will randomly sample
		#   from -- bigger means better shuffling but slower start up and more
		#   memory used.
		# capacity must be larger than min_after_dequeue and the amount larger
		#   determines the maximum we will prefetch.  Recommendation:
		#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
		min_after_dequeue = 50000
		capacity = min_after_dequeue + 3 * batch_size
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
		return example_batch, label_batch

	def read_format_sentsim(self,filename_queue):

		# file reader and value generator
		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)
		
		# default format of the file
		record_defaults = [[]] * ((self.max_length*2)+1)
		record_defaults[1].append(0)

		# extract all csv columns
		features = tf.decode_csv(value,record_defaults=record_defaults)

		# make different inputs for two sentences/examples
		example1 = features[:self.max_length]
		example2 = features[self.max_length:self.max_length*2]
		
		# extract label, whether they are similar/1 or not/0
		label = tf.to_float(features[-1])

		label = tf.stack([label])

		return example1, example2, label

	def input_pipeline_sentsim(self,filenames, batch_size, num_epochs=None):
		filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=num_epochs, shuffle=True)
		example1, example2, label = self.read_format_sentsim(filename_queue)
		# min_after_dequeue defines how big a buffer we will randomly sample
		#   from -- bigger means better shuffling but slower start up and more
		#   memory used.
		# capacity must be larger than min_after_dequeue and the amount larger
		#   determines the maximum we will prefetch.  Recommendation:
		#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
		min_after_dequeue = 100000
		capacity = min_after_dequeue + 3 * batch_size 
		example1_batch, example2_batch, label_batch = tf.train.shuffle_batch(
			[example1, example2, label], batch_size=batch_size, capacity=capacity,
			min_after_dequeue=min_after_dequeue,shapes=None)
		return example1_batch, example2_batch, label_batch

	def fit(self):

		# create a session
		coord = tf.train.Coordinator()

		self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.num_threads))

		# with self.sess as session:
		session = self.sess

		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		session.run(self.init_op)

		average_loss = 0
		sentsim_average_loss = 0

		# op to write logs to Tensorboard
		summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

		print("Initialized")

		for step in range(self.num_steps):

			#if we are doing joint training
			if self.joint == 'true':

				#run session

				# skip-gram step
				'''
				_, loss_val,summary = session.run([self.skip_optimizer, self.skip_loss,
					self.merged_summary_skip])
				'''
				doc1_batch, doc2_batch, doc3_batch, doc4_batch, seq_len_batch, doc_len_batch = self._generate_batch_docsim() 
				# create batches for sentence similarity task
				print(doc1_batch.shape)
				print (doc2_batch.shape)
				print (doc3_batch.shape)
				print (doc4_batch.shape)
				print (seq_len_batch.shape)
				print (doc_len_batch.shape)

				print(doc_len_batch)

				feed_dict = {self.doc1:doc1_batch, self.doc2:doc2_batch, self.doc3:doc3_batch, self.doc4:doc4_batch,
				self.seq_len: seq_len_batch, self.doc_len: doc_len_batch, self.keep_prob: 0.7}

				_, sent_loss, document_loss, totalsent_loss, summary_doc2vec = session.run([self.task_optimizer, self.totalsent_loss, self.document_loss,
				 self.totalsent_loss, self.merged_summary_task_mlp], feed_dict=feed_dict)
				print(sent_loss)
				print(document_loss)
				print(totalsent_loss)

				'''
				train_sentsimx_batch, train_sentsimy_batch, train_sentsim_labels_batch = session.run([self.train_sentsimx_batch,
					self.train_sentsimy_batch, self.train_sentsim_labels_batch])
				
				# run the sentence similarity task
				embedx,embedy = session.run([self.embedx, self.embedy],
					feed_dict={self.train_sentsimx: train_sentsimx_batch, self.train_sentsimy: train_sentsimy_batch,
					self.train_sentsim_labels: train_sentsim_labels_batch, self.keep_prob : 0.7 })

#				print embedx.shape
#				print embedy.shape

				contextx,contexty = session.run([self.contextx, self.contexty],
					feed_dict={self.train_sentsimx: train_sentsimx_batch, self.train_sentsimy: train_sentsimy_batch,
					self.train_sentsim_labels: train_sentsim_labels_batch, self.keep_prob : 0.7 })

#				print contextx.shape
#				print contexty.shape
				_, sentsim_loss, summary_sentsim = session.run([self.task_optimizer, self.cost_mean,self.merged_summary_task_mlp],
					feed_dict={self.train_sentsimx: train_sentsimx_batch, self.train_sentsimy: train_sentsimy_batch,
					self.train_sentsim_labels: train_sentsim_labels_batch, self.keep_prob : 0.7 })

				'''
				#add  loss summary at for skip-gram
				'''
				summary_writer.add_summary(summary, step)
				'''
				#add loss summary for doc2vec
				summary_writer.add_summary(summary_doc2vec, step)
#				summary_writer.add_summary(summary_sentsim, step)

			

				#add loss 
#				average_loss += loss_val
#				sentsim_average_loss += sentsim_loss

			
			if step % 1000 == 0:

				# read the validation data batch by batch and compute total accuracy
				total_valid_accuracy = 0

				# create batches for sentence similarity task

				'''
				valid_sentsimx_batch, valid_sentsimy_batch, valid_sentsim_labels_batch = session.run([self.valid_sentsimx_batch,
					self.valid_sentsimy_batch, self.valid_sentsim_labels_batch])


				cost_mean, cost_match_mean, cost_mismatch_mean = session.run([self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean], feed_dict={self.train_sentsimx: valid_sentsimx_batch,
					self.train_sentsimy: valid_sentsimy_batch, self.train_sentsim_labels: valid_sentsim_labels_batch,
					self.keep_prob : 0.7 })
				#valid_accuracy = self.acc.eval({self.train_sentsimx: valid_sentsimx_batch, 
				#		self.train_sentsimy: valid_sentsimy_batch, self.train_sentsim_labels: valid_sentsim_labels_batch,
				#		self.keep_prob: 1.0}, session=session)

				summary = tf.Summary(value=[tf.Summary.Value(tag="valid-loss",
					 simple_value=float(cost_mean))])

				summary_match = tf.Summary(value=[tf.Summary.Value(tag="valid-match-loss",
					 simple_value=float(cost_match_mean))])

				summary_mismatch = tf.Summary(value=[tf.Summary.Value(tag="valid-mismatch-loss",
					 simple_value=float(cost_mismatch_mean))])
					
				summary_writer.add_summary(summary, step)
				summary_writer.add_summary(summary_match, step)
				summary_writer.add_summary(summary_mismatch, step)
			
				'''
			if step % 500 == 0:
				if step > 0:
					average_loss /= 500
					sentsim_average_loss /= 500
#					task_mlp_average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print("Average loss of skip-gram step ", step, ": ", average_loss)
				print("Average loss of sent-sim at step ", step, ": ", sentsim_average_loss)
				average_loss = 0
				sentsim_average_loss = 0

			
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
