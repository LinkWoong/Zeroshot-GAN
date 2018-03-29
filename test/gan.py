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
import logging
logging.getLogger('tensorflow').disabled = True
#------------------------CL Setting-----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='./train_images/', help='Training directory')
parser.add_argument('--test_dir', default='./test_images/', help='Testing directory')
parser.add_argument('--save_dir', default='./save_images', help='Saving directory')
parser.add_argument('--training_epoch', default=20, type=int, help='Number of training epoches')
parser.add_argument('--is_training', default=True, type=bool, help='Status of training')
parser.add_argument('--is_testing', default=False, type=bool, help='Status of testing')
parser.add_argument('--batch_size', default=10, type=int, help='Number of batch_size')
parser.add_argument('--D_learning_rate', default=0.00001, type=float,help='Discriminator learning rate')
parser.add_argument('--G_learning_rate', default=0.00001, type=float,help='Generator learning rate')
parser.add_argument('--num_of_data', default=50000, type=int, help='Number of training examples')

#-----------------------Parameter Setting-----------------
#log_device_placement = True

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

OUTPUT_WIDTH = 32
OUTPUT_HEIGHT = 32

TRAIN_SAMPLES = 50000
TRAINING_EPOCH = 20
TEST_SAMPLES = 10000
NUM_OF_CLASSES = 10

#D_learning_rate = 0.0002
#G_learning_rate = 0.0002

D_iter = 5
G_iter = 1

batch_size = 10
gpu_options = True

training_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/train'
testing_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test'

training_image_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/train_image'
testing_image_path = '/media/linkwong/File/Ubuntus/cifar-10-batches-py/test_image'
save_dir = '/media/linkwong/File/Ubuntus/dump/'

#-----------------Some utilities----------------------------------------------------

def batch_normalization(x, is_training):
	return tf.layers.batch_normalization(x, training=is_training)

def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, name="conv2d"):
	#print x.get_shape()
	with tf.variable_scope(name):
		weight = tf.get_variable('w', shape=[k_h, k_w, x.get_shape()[-1], output_dim], 
								initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('b', shape=[output_dim], initializer=tf.constant_initializer(0.0))

		conv = tf.nn.conv2d(x, weight, strides=[1, d_h, d_w, 1], padding='SAME')
		weight_conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

	return weight_conv

def deconv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, name="deconv2d"):
	print x.get_shape()
	with tf.variable_scope(name):
		weight = tf.get_variable('w', shape=[k_h, k_w, output_dim[-1], x.get_shape()[-1]], 
								initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('b', shape=[output_dim[-1]], initializer=tf.constant_initializer(0.0))

		deconv = tf.nn.conv2d_transpose(x, weight, strides=[1, d_h, d_w, 1], output_shape=output_dim)
		weight_conv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

	return weight_conv

def linear(x, output_dim ,scope=None, stddev=0.02, bias_start=0.0):
	shape = x.get_shape().as_list()
	#print shape
	with tf.variable_scope(scope):
		matrix = tf.get_variable("matrix", shape=[shape[1], output_dim], dtype=tf.float32, 
								initializer=tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32,
								initializer=tf.constant_initializer(bias_start))

	return tf.matmul(x, matrix) + bias
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

	return tf.nn.leaky_relu(x + 1e09, alpha=alpha)

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


def prepare_input(data=None, labels=None):

	"""
	normalize
	"""
	image_height = 32
	image_width = 32
	image_depth = 3

	assert(data.shape[1] == image_height * image_width * image_depth)
	assert(data.shape[0] == labels.shape[0])

	mu = np.mean(data, axis=0)
	mu = mu.reshape(1,-1)

	sigma = np.std(data, axis=0)
	sigma = sigma.reshape(1, -1)
	data = data - mu
	data = data / sigma
	is_nan = np.isnan(data)
	is_inf = np.isinf(data)
	if np.any(is_nan) or np.any(is_inf):
	    print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
	    #data is transformed from (no_of_samples, 3072) to (no_of_samples , image_height, image_width, image_depth)
	    #make sure the type of the data is no.float32

	data = data.reshape([-1,image_depth, image_height, image_width])
	data = data.transpose([0, 2, 3, 1])
	data = data.astype(np.float32)
	noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)

	data = data + noise
	return data, labels

def unpickle(relpath): 
    with open(relpath, 'rb') as fp:
        d = pickle.load(fp)
    return d

def read_cifar10(filename): # queue one element

	class CIFAR10Record(object):
		pass
	result = CIFAR10Record()

	label_bytes = 1  # 2 for CIFAR-100
	result.height = 32
	result.width = 32
	result.depth = 3

	data = unpickle(filename)

	value = np.asarray(data.get('data')).astype(np.float32)
	labels = np.asarray(data.get('labels')).astype(np.int32)

	return prepare_input(value,labels)
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
		if self.is_training:
			train_data, train_label = self.load_raw_data()
			for i in range(len(train_data)):
				count = 0
				for item in train_data[i]:
					item = item.reshape(3, 32, 32)
					item = item.transpose(1, 2, 0)
					if not os.path.exists(self.training_path + 'train_image/'):
						os.mkdir(self.training_path + 'train_image/')
						misc.imsave(self.training_path + 'train_image/' + self.train_filenames[i][count], item)
						count += 1
			#print train_data[0][0].shape

		elif self.is_testing:
			train_data, train_label = self.load_raw_data()
			for i in range(len(train_data)):
				count = 0
				for item in train_data[i]:
					item = item.reshape(3, 32, 32)
					item = item.transpose(1, 2, 0)
					if not os.path.exists(self.testing_path + 'test_image/'):
						os.mkdir(self.testing_path + 'test_image/')
						misc.imsave(self.testing_path + 'test_image/' + self.train_filenames[i][count], item)
						count += 1

		return train_data, train_label, self.train_filenames

#------------------------------model-build---------------------------------------

class model(object):
	"""
	Implementation of model build
	"""

	def __init__(self, train_dir, test_dir, save_dir, training_epoch, is_training, is_testing, batch_size=10, D_learning_rate=0.1, 
																	G_learning_rate=0.1, num_of_data=50000, D_iter=5, G_iter=1, stddev=0.02, drop_out=0.3, alpha=0.2):
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.save_dir = save_dir
		self.training_epoch = training_epoch
		self.is_training = is_training
		self.is_testing = is_testing
		self.batch_size = batch_size
		self.D_learning_rate = D_learning_rate
		self.G_learning_rate = G_learning_rate
		self.D_iter = D_iter
		self.G_iter = G_iter
		self.stddev = stddev
		self.drop_out = drop_out
		self.alpha = alpha
		self.sess = tf.Session()
		self.num_of_data = num_of_data

	def load_data(self):
		"""
		Load data into tensors
		"""
		ass = data(self.train_dir, self.test_dir, is_training=self.is_training, is_testing=self.is_testing)
		train_data, train_label, train_filenames = ass.load_and_save_data()
		#print len(train_data), len(train_label), len(train_filenames) #5, 5, 5

		if self.is_training:
			new_train_filenames = []

			for i in range(len(train_filenames)):
				for item in train_filenames[i]:
					new_train_filenames.append(self.train_dir + '/train_image/' + item)

			#print len(new_train_filenames)

			tf_filenames = tf.constant(new_train_filenames)
			dataset = tf.data.Dataset.from_tensor_slices(tf_filenames)
			dataset = dataset.map(parse_file)

		elif self.is_testing:
			new_test_filenames = []

			for i in range(len(train_filenames)):
				for item in train_filenames[i]:
					new_test_filenames.append(self.test_dir + '/test_image/' + item)

			tf_filenames = tf.constant(new_test_filenames)
			dataset = tf.data.Dataset.from_tensor_slices(tf_filenames)
			dataset = dataset.map(parse_file)

		return dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.training_epoch)

	def load_test_data(self):
		"""
		Load the test 100 images for model testing
		"""
		test_path = '/home/chenhui/fyp/cifar-10/test_image/'

		filenames = []

		for i in os.listdir(test_path):
			filenames.append(os.path.join(test_path, i))

		print "Length of filenames is", len(filenames)
		tf_filenames = tf.constant(filenames)
		dataset = tf.data.Dataset.from_tensor_slices(tf_filenames)
		dataset = dataset.map(parse_file)

		return dataset

	def load_prepared_data(self):
		"""
		Load the prepared data
		"""
		data_dir = '/home/chenhui/fyp/cifar-10/train/'
		filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]

		for idx, filename in enumerate(filenames):
			temp_X, temp_y = read_cifar10(filename)
			if idx == 0:
				dataX = temp_X
				labely = temp_y
			else:
				dataX = np.append(dataX, temp_X)
				labely = np.append(labely, temp_y)
			dataX = dataX.reshape([-1, 32, 32, 3])

		seed = 547
		np.random.seed(seed)
		np.random.shuffle(dataX)
		np.random.seed(seed)
		np.random.shuffle(labely)

		y_vec = np.zeros((len(labely), 10), dtype=np.float)
		for i, label in enumerate(labely):
			y_vec[i, labely[i]] = 1.0

		return dataX / 255., y_vec


	def generator(self, x, is_training=True, reuse=False, scope='generator'):

		"""
		G(z) Implementation
		The generator that learns data distribution
		Use Deep Convolution Layers instead of FC layers

		"""
		print "Reaching generator"

		x = tf.convert_to_tensor(x, dtype=tf.float32)

		with tf.variable_scope('generator',reuse=reuse):

			#x = linear(x, 512*2*2, scope='g_fc_1')
			#x = tf.reshape(x, shape=[self.batch_size, 2, 2, 512])

			x_deconv_1 = linear(x, 512*2*2, scope='x_deconv_1')
			x_deconv_1 = tf.reshape(x_deconv_1, [self.batch_size, 2, 2, 512])
			x_deconv_1_ac = tf.layers.batch_normalization(x_deconv_1, training=self.is_training, name='x_deconv_1_bn')
			x_deconv_1_ac = tf.nn.leaky_relu(x_deconv_1_ac, name='x_deconv_1_ac')

			x_deconv_2 = deconv2d(x_deconv_1_ac, [self.batch_size, 4, 4, 256], 5, 5, 2, 2, name='x_deconv_2')
			x_deconv_2_ac = tf.layers.batch_normalization(x_deconv_2, training=self.is_training, name='x_deconv_2_bn')
			x_deconv_2_ac = tf.nn.leaky_relu(x_deconv_2_ac, name='x_deconv_2_ac')

			x_deconv_3 = deconv2d(x_deconv_2_ac, [self.batch_size, 8, 8, 128], 5, 5, 2, 2, name='x_deconv_3')
			x_deconv_3_ac = tf.layers.batch_normalization(x_deconv_3, training=self.is_training, name='x_deconv_3_bn')
			x_deconv_3_ac = tf.nn.leaky_relu(x_deconv_3_ac, name='x_deconv_3_ac')

			x_deconv_4 = deconv2d(x_deconv_3_ac, [self.batch_size, 16, 16, 64], 5, 5, 2, 2, name='x_deconv_4')
			x_deconv_4_ac = tf.layers.batch_normalization(x_deconv_4, training=self.is_training, name='x_deconv_4_bn')
			x_deconv_4_ac = tf.nn.leaky_relu(x_deconv_4_ac, name='x_deconv_4_ac')

			x_out = deconv2d(x_deconv_4_ac, [self.batch_size, 32, 32, 3], 5, 5, 2, 2, name='x_deconv_5')
			x_out = tf.nn.tanh(x_out, name='x_out')


		return x_out

	def discriminator(self, x, is_training=True, reuse=False, scope='discriminator'):
		"""
		D(x) implementation
		The discriminator that compare G(z) with true data distribution P(x)
		Use Deep Convolutional Layers
		"""
		print "Reaching discriminator"

		with tf.variable_scope('discriminator', reuse=reuse):
			print "The shape of generated is ", x.get_shape()
			x_conv_1 = conv2d(x, 64, 5, 5, 2, 2, name='x_conv_1')
			x_conv_1_ac = tf.nn.leaky_relu(x_conv_1, name='x_conv_1_ac')

			x_conv_2 = conv2d(x_conv_1_ac, 128, 5, 5, 2, 2, name='x_conv_2')
			x_conv_2_ac = tf.layers.batch_normalization(x_conv_2, training=self.is_training, name='x_conv_2_bn')
			x_conv_2_ac = tf.nn.leaky_relu(x_conv_2_ac, name='x_conv_2_ac')

			x_conv_3 = conv2d(x_conv_2_ac, 256, 5, 5, 2, 2, name='x_conv_3')
			x_conv_3_ac = tf.layers.batch_normalization(x_conv_3, training=self.is_training, name='x_conv_3_bn')
			x_conv_3_ac = tf.nn.leaky_relu(x_conv_3_ac, name='x_conv_3_ac')

			x_conv_4 = conv2d(x_conv_3_ac, 512, 5, 5, 2, 2, name='x_conv_4')
			x_conv_4_ac = tf.layers.batch_normalization(x_conv_4, training=self.is_training, name='x_conv_4_bn')
			x_conv_4_ac = tf.nn.leaky_relu(x_conv_4_ac, name='x_conv_4_ac')

			x_conv_4_ac = tf.reshape(x_conv_4_ac, [self.batch_size, -1])

			x_out_logit = linear(x_conv_4_ac, 1, scope='x_out_logit')
			x_out = tf.nn.sigmoid(x_out_logit)

		return x_out, x_out_logit ,x_conv_4_ac

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
		with tf.Graph().as_default():

			train_data, train_label = self.load_prepared_data()

			print "The shape of train_data is ", train_data.shape
			#train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.training_epoch)
			#iterator = train_dataset.make_one_shot_iterator()
			#one_element = iterator.get_next()

			noise_limit = 0.35
			L2_PENALTY = 0.02

			x = tf.placeholder(tf.float32, shape=(self.batch_size, 32, 32, 3)) # training examples
			z = tf.placeholder(tf.float32, shape=(self.batch_size, 100)) # Noises that feed to the generator 
			is_training = tf.placeholder(tf.bool)

			G_z = self.generator(z, is_training=is_training)
			D_real, D_real_logits, _ = self.discriminator(x, is_training=is_training) # D_real: the real convoluted training examples 
																				 # D_real_logit: the convoluted but not activated training examples
			D_fake, D_fake_logits, _ = self.discriminator(G_z, is_training=is_training, reuse=True)

			D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
			D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

			D_loss_total = D_loss_real + D_loss_fake

			G_loss_total = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

			T_vars = tf.trainable_variables()
			D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
			G_vars = [var for var in T_vars if var.name.startswith('generator')]
			
			D_optimize = tf.train.AdamOptimizer(self.D_learning_rate, beta1=0.5).minimize(D_loss_total, var_list=D_vars)
			G_optimize = tf.train.AdamOptimizer(self.G_learning_rate, beta1=0.5).minimize(G_loss_total, var_list=G_vars)

			D_loss_real_sum = tf.summary.scalar("D_loss_real", D_loss_real)
			D_loss_fake_sum = tf.summary.scalar("D_loss_fake", D_loss_fake)
			D_loss_total_sum = tf.summary.scalar("D_loss_total", D_loss_total)
			G_loss_total_sum =  tf.summary.scalar("G_loss_total", G_loss_total)

			G_sum = tf.summary.merge([D_loss_fake_sum, G_loss_total_sum])
			D_sum = tf.summary.merge([D_loss_real_sum, D_loss_total_sum])
			#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
			sess = tf.Session()

			#self.sess = sess
			#--------------------------------Model-Training-------------------------------------------------------------
			sess.run(tf.global_variables_initializer())

			writer = tf.summary.FileWriter('/home/chenhui/fyp/logs/gan', sess.graph)
			train_history = {}
			train_history['D_losses'] = []
			train_history['G_losses'] = []

			count = 1
			fixed_z = np.random.normal(-1, 1, [self.batch_size, 100]) #Fixed Z for feeding

			for i in range(self.training_epoch):

				D_losses = []
				G_losses = []

				print "Current epoch", i
				#print "Start building the model with dataset length", num_of_data

				current_epoch = i

	 			for j in range(0, len(train_data) / self.batch_size):
					#print "Current inter-index", j
					batch_images = train_data[j*self.batch_size:(j+1)*self.batch_size]
					#assert not np.any(np.isnan(batch_images)), "Error! The array becomes NaN"
					#print temp.shape
					z_ = np.random.normal(-1, 1, [self.batch_size, 100])
					loss_d, _, summary_str = sess.run([D_loss_total, D_optimize, D_sum], feed_dict={x: batch_images, z: z_, is_training: True})
					writer.add_summary(summary_str, count)
					D_losses.append(loss_d)

					count += 1
					#z_ = np.random.normal(0, 1, [self.batch_size, 100])
					loss_g, _, summary_str  = sess.run([G_loss_total, G_optimize, G_sum], feed_dict={x: batch_images, z: z_, is_training: True})
					writer.add_summary(summary_str, count)
					G_losses.append(loss_g)

					if j % 50 == 0:

						test_image = sess.run(G_z, feed_dict={z: fixed_z, is_training: True})
						# print "The test image shape is ", test_image.shape (100, 32, 32, 3)
						random_index = np.random.random_integers(0, self.batch_size - 1)
						temp_dir = self.save_dir + '/result_' + str(i) + '_' + str(j) + '.png'
						self.save_image(test_image[self.batch_size - 1].reshape(32, 32, 3), temp_dir)
				
				train_history['D_losses'].append(np.mean(D_losses))
				train_history['G_losses'].append(np.mean(G_losses))
				print "At epoch " + str(i) + " Current D_losses are ", str(np.mean(train_history.get('D_losses')))
				print "At epoch " + str(i) + " Current G_losses are ", str(np.mean(train_history.get('G_losses')))

		return train_history

def main(argv):
	"""
	Main function
	"""
	args = parser.parse_args(argv[1:])
	ass = model(train_dir=args.train_dir, test_dir=args.test_dir, save_dir=args.save_dir, training_epoch=args.training_epoch,
			batch_size=args.batch_size, is_training=args.is_training, is_testing=args.is_testing, D_learning_rate=args.D_learning_rate,
			G_learning_rate=args.G_learning_rate, num_of_data=args.num_of_data)
	result = ass.build_model()

if __name__ == '__main__':
	#tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)