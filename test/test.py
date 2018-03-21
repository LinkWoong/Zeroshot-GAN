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
import random

#------------------------CL Setting-----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='./train_images/', help='Training directory')
parser.add_argument('--test_dir', default='./test_images/', help='Testing directory')
parser.add_argument('--save_dir', default='./save_images', help='Saving directory')
parser.add_argument('--training_epoch', default=20, help='Number of training epoches')
parser.add_argument('--is_training', default=1, help='Status of training')
parser.add_argument('--is_testing', default=0, help='Status of testing')
parser.add_argument('--batch_size', default=10, help='Number of batch_size')
parser.add_argument('--D_learning_rate', default=0.0002, help='Discriminator learning rate')
parser.add_argument('--G_learning_rate', default=0.0002, help='Generator learning rate')

#-----------------------Parameter Setting-----------------

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

OUTPUT_WIDTH = 32
OUTPUT_HEIGHT = 32

TRAIN_SAMPLES = 50000
TRAINING_EPOCH = 20
TEST_SAMPLES = 10000
NUM_OF_CLASSES = 10

D_learning_rate = 0.0002
G_learning_rate = 0.0002

D_iter = 5
G_iter = 1

batch_size = 10

training_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/train'
testing_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test'

training_image_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/train_image'
testing_image_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test_image'
save_dir = '/media/linkwong/File/Ubuntus/dump/'

#-----------------Some utilities----------------------------------------------------

def batch_normalization(x, is_training, epsilon=0.001, momentum=0.99, axis=-1):
	return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=is_training, axis=axis)

def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, scope="conv2d"):
	shape = x.get_shape().as_list()
	print "The shape is ", shape

	with tf.variable_scope(scope):
		weight = tf.get_variable('w', shape=[k_h, k_w, shape[-1], output_dim], 
								initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('w', shape=[output_dim], initializer=tf.constant_initializer(0.0))

		conv = tf.layers.conv2d(x, weight, strides=[1, d_h, d_w, 1], padding='SAME')
		weight_conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

	return weight_conv

def deconv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, scope="deconv2d"):
	shape = x.get_shape().as_list()
	print "The shape is ", shape

	with tf.variable_scope(scope):
		weight = tf.get_variable('w', shape=[k_h, k_w, output_dim[-1], shape[-1]], 
								initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('w', shape=[output_dim[-1]], initializer=tf.constant_initializer(0.0))

		deconv = tf.nn.conv2d_transpose(x, weight, strides=[1, d_h, d_w, 1], output_shape=output_dim)
		weight_conv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

	return weight_conv

def parse_file(filenames):
	"""
	Normalize the image between -1 and 1
	"""
	img_string = tf.read_file(filenames)
	img_decode = tf.image.decode_image(img_string)
	img_cast = tf.cast(img_decode, tf.float32)

	mean, var = tf.nn.moments(img_cast, axes=[1])
	img_normal = (tf.image.per_image_standardization(img_decode) - mean) / var # Centralization

	return img_normal

def leaky_relu(x, alpha=0.2):

	return tf.nn.leaky_relu(x, alpha=alpha)

def unit_test(dataset):
	count = 0
	iterator = dataset.make_one_shot_iterator()
	one_element = iterator.get_next()
	with tf.Session() as sess:
		try:
			while True:
				temp = sess.run(one_element)
				count += 1
		except tf.errors.OutOfRangeError:
			print "End!"

	return count

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

#------------------------------model-build---------------------------------------

class model(object):
	"""
	Implementation of model build
	"""

	def __init__(self, train_dir, test_dir, save_dir, training_epoch, is_training, is_testing, batch_size=10, D_learning_rate=0.0002, 
																	G_learning_rate=0.0002, D_iter=5, G_iter=1, stddev=0.02, drop_out=0.3, alpha=0.2):
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.save_dir = save_dir
		self.training_epoch = training_epoch
		self.is_training = is_training
		self.is_testing = is_testing
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.D_iter = D_iter
		self.G_iter = G_iter
		self.stddev = stddev
		self.drop_out = drop_out
		self.alpha = alpha

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

		return dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(10)

	def load_test_data(self):
		"""
		Load the test 100 images for model testing
		"""
		test_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test_image/'

		filenames = []

		for i in os.listdir(test_path):
			filenames.append(os.path.join(test_path, i))

		print "Length of filenames is", len(filenames)
		tf_filenames = tf.constant(filenames)
		dataset = tf.data.Dataset.from_tensor_slices(tf_filenames)
		dataset = dataset.map(parse_file)

		return dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(10)

	def generator(self, x, is_training=True, reuse=False, scope='generator'):

		"""
		G(z) Implementation
		The generator that learns data distribution
		Use Deep Convolution Layers instead of FC layers

		"""
		x = tf.convert_to_tensor(x, dtype=tf.float32)

		with tf.variable_scope('generator',reuse=reuse):

			x_deconv_1 = tf.layers.conv2d_transpose(inputs=x, filters=256, kernel_size=[2, 2], strides=(1, 1), padding='valid')
			x_deconv_1_ac = leaky_relu(batch_normalization(x_deconv_1, is_training=is_training), alpha=self.alpha)

			x_deconv_2 = tf.layers.conv2d_transpose(inputs=x_deconv_1_ac, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_deconv_2_ac = leaky_relu(batch_normalization(x_deconv_2, is_training=is_training), alpha=self.alpha)

			x_deconv_3 = tf.layers.conv2d_transpose(inputs=x_deconv_2_ac, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_deconv_3_ac = leaky_relu(batch_normalization(x_deconv_3, is_training=is_training), alpha=self.alpha)

			x_deconv_4 = tf.layers.conv2d_transpose(inputs=x_deconv_3_ac, filters=32, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_deconv_4_ac = leaky_relu(batch_normalization(x_deconv_4, is_training=is_training), alpha=self.alpha)

			x_deconv_5 = tf.layers.conv2d_transpose(inputs=x_deconv_4_ac, filters=3, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_out = tf.nn.tanh(x_deconv_5)
		#print x_out.get_shape()
		return x_out

	def discriminator(self, x, is_training=True, reuse=False, scope='discriminator'):
		"""
		D(x) implementation
		The discriminator that compare G(z) with true data distribution P(x)
		Use Deep Convolutional Layers
		"""
		x = tf.convert_to_tensor(x, dtype=tf.float32)

		with tf.variable_scope('discriminator', reuse=reuse):

			x_conv_1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_conv_1_ac = leaky_relu(x_conv_1, alpha=self.alpha)

			x_conv_2 = tf.layers.conv2d(inputs=x_conv_1_ac, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_conv_2_ac = leaky_relu(batch_normalization(x_conv_2, is_training=is_training), alpha=self.alpha)

			x_conv_3 = tf.layers.conv2d(inputs=x_conv_2_ac, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_conv_3_ac = leaky_relu(batch_normalization(x_conv_3, is_training=is_training), alpha=self.alpha)

			x_conv_4 = tf.layers.conv2d(inputs=x_conv_3_ac, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='same')
			x_conv_4_ac = leaky_relu(batch_normalization(x_conv_4, is_training=is_training), alpha=self.alpha)

			x_conv_5 = tf.layers.conv2d(inputs=x_conv_4_ac, filters=3, kernel_size=[2, 2], strides=(1, 1), padding='valid')
			x_out = tf.nn.sigmoid(x_conv_5)

		return x_out, x_conv_5

	def save_image(self, x, path):
		"""
		Save trained image for each epoch(would be 20)
		"""
		misc.imsave(path, x)
		pass

	def build_model(self):
		"""
		GAN model build and loss functions set up
		"""

		train_dataset = self.load_test_data()
		iterator = train_dataset.make_one_shot_iterator()
		one_element = iterator.get_next()

		num_of_data = unit_test(train_dataset)
		x = tf.placeholder(tf.float32, shape=(self.batch_size, 32, 32, 3)) # training examples
		z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100)) # Noises that feed to the generator 
		is_training = tf.placeholder(tf.bool)

		G_z = ass.generator(z, is_training=is_training)
		D_real, D_real_logits = ass.discriminator(x, is_training=is_training) # D_real: the real convoluted training examples 
																			 # D_real_logit: the convoluted but not activated training examples
		D_fake, D_fake_logits = ass.discriminator(G_z, is_training=is_training, reuse=True)

		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([self.batch_size, 1, 1, 3]), logits=D_real_logits))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([self.batch_size, 1, 1, 3]), logits=D_fake_logits))

		D_loss_total = D_loss_real + D_loss_fake

		G_loss_total = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([self.batch_size, 1, 1, 3]), logits=D_fake_logits))

		T_vars = tf.trainable_variables()
		D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
		G_vars = [var for var in T_vars if var.name.startswith('generator')]

		D_optimize = tf.train.AdamOptimizer(D_learning_rate, beta1=0.5).minimize(D_loss_total)
		G_optimize = tf.train.AdamOptimizer(G_learning_rate, beta1=0.5).minimize(G_loss_total)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		train_history = {}
		train_history['D_losses'] = []
		train_history['G_losses'] = []

		count = 0
		fixed_z = np.random.normal(0, 1, (self.batch_size, 1, 1, 100)) #Fixed Z for feeding

		for i in range(self.training_epoch):

			D_losses = []
			G_losses = []

			current_epoch = i

			for j in range(num_of_data / self.batch_size):

				temp = sess.run(one_element)
				z_ = np.random.normal(0, 1, (self.batch_size, 1, 1, 100))
				loss_d, _ = sess.run([D_loss_total, D_optimize], feed_dict={x: temp, z: z_, is_training: True})

				D_losses.append(loss_d)

				z_ = np.random.normal(0, 1, (self.batch_size, 1, 1, 100))
				loss_g, _ = sess.run([G_loss_total, G_optimize], feed_dict={x: temp, z: z_, is_training: True})

				G_losses.append(loss_g)

				test_image = sess.run(G_z, feed_dict={z: fixed_z, is_training: False})
				
				random_index = np.random.random_integers(0, self.batch_size)

				if(i == current_epoch):
					temp_dir = self.save_dir + 'result_' + str(current_epoch) + '.png'
					self.save_image(test_image[random_index], temp_dir)
					current_epoch = 0

			print "Current epoch", i
			train_history['D_losses'].append(np.mean(D_losses))
			train_history['G_losses'].append(np.mean(G_losses))
			print "Current D_losses", train_history.get('D_losses')
			print "Current G_losses", train_history.get('G_losses')

		if (i < (num_of_data / self.batch_size)):
			print "Epoch number is smaller than batch_size"

		return train_history

def main(argv):
	"""
	Main function
	"""
	args = parser.parse_args(argv[1:])
	ass = model(train_dir=args.train_dir, test_dir=args.test_dir, save_dir=args.save_dir, training_epoch=args.training_epoch,
			batch_size=args.batch_size, is_training=args.is_training, is_testing=args.is_testing)
	result = ass.build_model()
