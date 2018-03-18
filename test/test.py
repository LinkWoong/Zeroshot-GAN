# encoding = utf-8
# GAN Investigation on Cifar-10
# Date: 17th, Mar, 2018
# Title: Implementation of GAN in order to strengthen the comprehension

import numpy as np
import tensorflow as tf

import argparse
import os
import pickle
import scipy.misc as misc


#-----------------------Parameter Setting-----------------

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

TRAIN_SAMPLES = 50000
TEST_SAMPLES = 10000
NUM_OF_CLASSES = 10

D_learning_rate = 0.001
G_learning_rate = 0.001

D_iter = 5
G_iter = 1

batch_size = 100

training_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/train'
testing_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test'

training_image_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/train_image'
testing_image_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test_image'

#-----------------Some utilities----------------------------------------------------

def batch_normalization(x, is_training, scope, epsilon=0.001, momentum=0.99, axis=-1):
	return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=is_training, name=scope, axis=axis)

def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, scope="conv2d"):
	shape = x.get_shape().as_list()
	print "The shape is ", shape

	with tf.variable_scope(scope):
		weight = tf.get_variable('w', shape=[k_h, k_w, shape[-1], output_dim], 
								initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('w', shape=[output_dim], initializer=tf.constant_initializer(0.0))

		conv = tf.layers.conv2d(x, weight, strides=[1, d_h, d_w, 1], padding='SAME')
		add_bias = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

	return add_bias

def deconv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, scope="deconv2d"):
	shape = x.get_shape().as_list()
	print "The shape is ", shape

	with tf.variable_scope(scope):
		weight = tf.get_variable('w', shape=[k_h, k_w, output_dim[-1], shape[-1]], 
								initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('w', shape=[output_dim[-1]], initializer=tf.constant_initializer(0.0))

		deconv = tf.nn.conv2d_transpose(x, weight, strides=[1, d_h, d_w, 1], output_shape=output_dim)
		add_bias = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

	return add_bias

def parse_file(filenames):
	"""
	Normalize the image between -1 and 1
	"""
	img_string = tf.read_file(filenames)
	img_decode = tf.image.decode_image(img_string)
	img_cast = tf.cast(img_decode, tf.float32)

	mean, var = tf.nn.moments(img_cast, axes=[1])
	img_normal = (tf.image.per_image_standardization(img_decode) - mean) / var

	return img_normal
#---------------------Data Handling-----------------------

class data(object):
	"""
	Turn raw data into python-friendly (CIFAR-10 based)
	"""
	def __init__(self, training_path, testing_path, is_training=1, is_testing=0):

		self.training_path = training_path
		self.testing_path = testing_path
		self.is_training = is_training
		self.is_testing = is_testing

		self.train_data = []
		self.train_label = []
		self.train_filenames = []

		self.test_data = []
		self.test_label = []
		self.test_filenames = []

	def load_raw_data(self):
		"""
		Load the data into np array.
		Note that the data is np array and the label is list

		"""
		if(self.is_training):

			print "Loading the training set..."
			for i in os.listdir(self.training_path):
				with open(os.path.join(self.training_path, i), 'rb') as f:
					data_dict = pickle.load(f)
					train_data = data_dict.get('data')
					train_label = data_dict.get('labels')
					train_filenames = data_dict.get('filenames')

					# print type(train_data) -> nparray
					# print type(train_label) -> list
					self.train_data.append(train_data)
					self.train_label.append(train_label)
					self.train_filenames.append(train_filenames)


			return self.train_data, self.train_label

		if(self.is_testing):

			for i in os.listdir(self.testing_path):
				with open(os.path.join(self.testing_path, i), 'rb') as ff:
					data_dict = pickle.load(ff)
					test_data = data_dict.get('data')
					test_label = data_dict.get('labels')
					test_filenames = data_dict.get('filenames')

					self.test_data.append(test_data)
					self.test_label.append(test_label)
					self.test_filenames.append(test_filenames)

			return self.test_data, self.test_label

	def reconstruct_img(self):
		"""
		Trying to reconstruct the image
		"""
		print "Trying to reconstruct the images..."
		print self.train_data[0].shape
		print self.train_data[0][0].shape
		assert self.train_data[0][0].shape[0] == IMAGE_HEIGHT * IMAGE_WIDTH * 3, "Size failed"

		data_reconstruct = self.train_data[0][0].reshape(3, 32, 32)
		data_reconstruct = data_reconstruct.transpose(1, 2, 0)
		misc.imsave('/home/linkwong/Desktop/fuck.png', data_reconstruct)

		pass

	def load_and_save_data(self):
		"""
		Load the data and store into imgs
		"""
		train_data, train_label = self.load_raw_data()
		for i in range(len(train_data)):
			count = 0
			for item in train_data[i]:
				item = item.reshape(3, 32, 32)
				item = item.transpose(1, 2, 0)
				if not os.path.exists('/media/linkwong/File/Ubuntus/cifar-10-batches-py/train_image'):
					misc.imsave('/media/linkwong/File/Ubuntus/cifar-10-batches-py/train_image/'+ self.train_filenames[i][count], item)
					count += 1
		#print train_data[0][0].shape
		return train_data, train_label, self.train_filenames

class model(object):
	"""
	Implementation of model build
	"""

	def __init__(self, train_dir, test_dir, is_training, is_testing, batch_size=10, learning_rate=0.01, 
																					D_iter=5, G_iter=1):
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.is_training = is_training
		self.is_testing = is_testing
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.D_iter = D_iter
		self.G_iter = G_iter

	def load_data(self):
		"""
		Load data into tensors
		"""
		ass = data(self.train_dir, self.test_dir, is_training=1)
		train_data, train_label, train_filenames = ass.load_and_save_data()
		#print len(train_data), len(train_label), len(train_filenames) #5, 5, 5

		new_train_filenames = []

		for i in range(len(train_filenames)):
			for item in train_filenames[i]:
				new_train_filenames.append('/media/linkwong/File/Ubuntus/cifar-10-batches-py/train_image/' + item)

		#print len(new_train_filenames)

		tf_filenames = tf.constant(new_train_filenames)
		dataset = tf.data.Dataset.from_tensor_slices(tf_filenames)
		dataset = dataset.map(parse_file)

		return dataset

ass = model(training_path, testing_path, is_training=1, is_testing=0)
train_dataset = ass.load_data()

iterator = train_dataset.make_one_shot_iterator()
one_element = iterator.get_next()

with tf.Session() as sess:
	for i in range(1, 5):
		temp = sess.run(one_element)
		print type(temp)
		print temp.shape