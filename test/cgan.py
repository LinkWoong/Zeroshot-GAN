# encoding = utf-8
# cGAN Investigation on Fashion-MNIST
# Date: 2nd, April 2018
# Title: Implementation of conditional GAN

import numpy as np
import tensorflow as tf

import argparse
import os
import gzip
import pickle
import imageio
import random
import logging
import skimage

from sklearn.preprocessing import MinMaxScaler

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

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

OUTPUT_WIDTH = 28
OUTPUT_HEIGHT = 28

TRAIN_DATA = 60000
TRAIN_LABELS = 60000

TEST_DATA = 10000
TEST_LABELS = 10000
NUM_OF_CLASSES = 10 # 0-9

D_learning_rate = 0.0002
G_learning_rate = 0.0002

gpu_options = True

training_path = '/media/linkwong/File/Ubuntus/Fashion-Mnist/train/'
testing_path = '/media/linkwong/File/Ubuntus/Fashion-Mnist/test/'

#-----------------Some utilities----------------------------------------------------

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def batch_normalization(x, is_training):
	"""
	Batch normalization
	"""
	return tf.layers.batch_normalization(x, training=is_training)

def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, name="conv2d"):
	"""
	Standard conv
	"""
	#print x.get_shape()
	with tf.variable_scope(name):
		weight = tf.get_variable('w', shape=[k_h, k_w, x.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('b', shape=[output_dim], initializer=tf.constant_initializer(0.0))

		conv = tf.nn.conv2d(x, weight, strides=[1, d_h, d_w, 1], padding='SAME')
		weight_conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

	return weight_conv

def deconv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.01, name="deconv2d"):
	"""
	Standard deconv
	"""
	#print x.get_shape()
	with tf.variable_scope(name):
		weight = tf.get_variable('w', shape=[k_h, k_w, output_dim[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
		bias = tf.get_variable('b', shape=[output_dim[-1]], initializer=tf.constant_initializer(0.0))

		deconv = tf.nn.conv2d_transpose(x, weight, strides=[1, d_h, d_w, 1], output_shape=output_dim)
		weight_conv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

	return weight_conv

def linear(x, output_dim ,scope=None, stddev=0.02, bias_start=0.0):
	"""
	Standard FC
	"""
	shape = x.get_shape().as_list()
	#print shape
	with tf.variable_scope(scope):
		matrix = tf.get_variable("matrix", shape=[shape[1], output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32,
								initializer=tf.constant_initializer(bias_start))

	return tf.matmul(x, matrix) + bias

def leaky_relu(x, alpha=0.2):
	"""
	Standard LR
	"""
	return tf.nn.leaky_relu(x, alpha=alpha)

def conv_cond_concat(x, y):
	"""
	Concatenate the images with conditions
	"""
	return tf.concat([x, y*tf.ones([x.get_shape()[0], x.get_shape()[1], x.get_shape()[2], y.get_shape()[3]])], 3)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def preprocessing(data):
	"""
	Normalize the input images between -1 and 1
	"""
	data = np.expand_dims(data, axis=1)
	data = np.reshape(data, (28, 28, 1))
	assert data.shape == (28, 28, 1), "Error! The shape is not correct"
	scaler = MinMaxScaler(feature_range=(-1, 1))
	temp_1 = data[:, :, 0]
	temp_2 = data[:, :, 1]
	temp_3 = data[:, :, 2]

	result_1 = scaler.fit_transform(temp_1)
	result_2 = scaler.fit_transform(temp_2)
	result_3 = scaler.fit_transform(temp_3)

	result = np.dstack((result_1, result_2, result_3))

	result = np.reshape(result, (IMAGE_WIDTH*IMAGE_HEIGHT))

	return result

def reshape_shuffle_image(train_data, train_label, test_data, test_label):
	"""
	Reshape (batch_size, 784) to (batch_size, 28, 28, 1)
	"""
	train_data = train_data.reshape(60000, 28, 28)
	train_data = np.expand_dims(train_data, axis=3)
	print "The training data shape is", train_data.shape

	test_data = test_data.reshape(10000, 28, 28)
	test_data = np.expand_dims(test_data, axis=3)
	print "The testing data shape is", test_data.shape

	X = np.concatenate((train_data, test_data), axis=0)
	Y = np.concatenate((train_label, test_label), axis=0)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(Y)

	Y_vec = np.zeros((len(Y), 10), dtype=np.float)
	for i, label in enumerate(Y):
		Y_vec[i, Y[i]] = 1.0

	return X / 255., Y_vec

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

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

		self.y_dim = 10 # dimension of conditional vector (labels used in here)
		self.c_dim = 1 # dimension of generated images (28, 28, 1)


	def load_prepared_data(self):
		"""
		Load the prepared data
		"""

		x_train, y_train = load_mnist(self.train_dir, kind='train')
		x_test, y_test = load_mnist(self.test_dir, kind='t10k')

		return reshape_shuffle_image(x_train, y_train, x_test, y_test)

	def generator(self, x, y, is_training=True, reuse=False, scope='generator'):

		"""
		G(z) Implementation
		The generator that learns data distribution
		Use Deep Convolution Layers instead of FC layers

		"""
		print "Reaching generator"

		x = tf.convert_to_tensor(x, dtype=tf.float32)
		x = tf.concat([x, y], 1)


		with tf.variable_scope('generator',reuse=reuse):

			#x = linear(x, 512*2*2, scope='g_fc_1')
			#x = tf.reshape(x, shape=[self.batch_size, 2, 2, 512])

			x_deconv_1 = linear(x, 1024, scope='x_fc_1')
			#x_deconv_1 = tf.reshape(x_deconv_1, [self.batch_size, 2, 2, 512])
			x_deconv_1_ac = tf.layers.batch_normalization(x_deconv_1, training=self.is_training, name='x_fc_1_bn')
			x_deconv_1_ac = tf.nn.leaky_relu(x_deconv_1, name='x_fc_1_ac')

			x_deconv_2 = linear(x_deconv_1_ac, 128 * 7 * 7, scope='x_fc_2')
			x_deconv_2_ac = tf.layers.batch_normalization(x_deconv_2, epsilon=1e-5, momentum=0.9, axis=-1, training=self.is_training, name='x_fc_2_bn')
			x_deconv_2_ac = tf.nn.leaky_relu(x_deconv_2_ac, name='x_fc_2_ac')

			x_deconv_2_ac = tf.reshape(x_deconv_2_ac, [self.batch_size, 7, 7, 128])

			x_deconv_3 = deconv2d(x_deconv_2_ac, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='x_deconv_3')
			x_deconv_3_ac = tf.layers.batch_normalization(x_deconv_3, epsilon=1e-5, momentum=0.9, axis=-1, training=self.is_training, name='x_deconv_3_bn')
			x_deconv_3_ac = tf.nn.leaky_relu(x_deconv_3_ac, name='x_deconv_3_ac')

			x_deconv_4 = deconv2d(x_deconv_3_ac, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='x_deconv_4')
			#x_deconv_4_ac = tf.layers.batch_normalization(x_deconv_4, epsilon=1e-5, momentum=0.9, axis=-1, training=self.is_training, name='x_deconv_4_bn')
			#x_deconv_4_ac = tf.nn.leaky_relu(x_deconv_4_ac, name='x_deconv_4_ac')

			#x_out = deconv2d(x_deconv_4_ac, [self.batch_size, 32, 32, 3], 5, 5, 2, 2, name='x_deconv_5')
			x_out = tf.nn.tanh(x_deconv_4, name='x_out')
			print x_out.get_shape()


		return x_out

	def discriminator(self, x, y, is_training=True, reuse=False, scope='discriminator'):
		"""
		D(x) implementation
		The discriminator that compare G(z) with true data distribution P(x)
		Use Deep Convolutional Layers
		x -> images
		y -> conditions (labels)
		"""
		print "Reaching discriminator"

		with tf.variable_scope('discriminator', reuse=reuse):
			#print "The shape of generated is ", x.get_shape()

			y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
			x = conv_cond_concat(x, y)

			x_conv_1 = conv2d(x, 64, 4, 4, 2, 2, name='x_conv_1')
			x_conv_1_ac = tf.nn.leaky_relu(x_conv_1, name='x_conv_1_ac')

			x_conv_2 = conv2d(x_conv_1_ac, 128, 4, 4, 2, 2, name='x_conv_2')
			x_conv_2_ac = tf.layers.batch_normalization(x_conv_2, epsilon=1e-5, momentum=0.9, axis=-1, training=self.is_training, name='x_conv_2_bn')
			x_conv_2_ac = tf.nn.leaky_relu(x_conv_2_ac, name='x_conv_2_ac')

			x_conv_2_ac = tf.reshape(x_conv_2_ac, [self.batch_size, -1])

			x_conv_3 = linear(x_conv_2_ac, 1024, scope='x_fc_3')
			x_conv_3_ac = tf.layers.batch_normalization(x_conv_3, epsilon=1e-5, momentum=0.9, axis=-1, training=self.is_training, name='x_conv_3_bn')
			x_conv_3_ac = tf.nn.leaky_relu(x_conv_3_ac, name='x_conv_3_ac')

			x_out_logit = linear(x_conv_3_ac, 1, scope='x_fc_out')
			x_out = tf.nn.tanh(x_out_logit)

		return x_out, x_out_logit ,x_conv_3_ac

	def save_image(self, x, path):
		"""
		Save trained image for each epoch
		"""
		imageio.imwrite(path, x)	
		pass

	def visualize_results(self, epoch, sess):
		"""
		Visualize the results
		"""

		tot_num_samples = self.batch_size
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
		y = np.random.choice(self.y_dim, self.batch_size)
		y_one_hot = np.zeros((self.batch_size, self.y_dim))

		y_one_hot[np.arange(self.batch_size), y] = 1

		z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 100))
		samples = sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})

		save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], image_path=self.save_dir + '/result_' + str(epoch) + '.png')
		n_styles = 10  

		np.random.seed()
		si = np.random.choice(self.batch_size, n_styles)

		for l in range(self.y_dim):
			y = np.zeros(self.batch_size, dtype=np.int64) + l

			y_one_hot = np.zeros((self.batch_size, self.y_dim))
			y_one_hot[np.arange(self.batch_size), y] = 1
			samples = sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})

			save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], image_path=self.save_dir + '/result_' + str(epoch) + '_' + str(l) + '.png')

			samples = samples[si, :, :, :]

			if l == 0:
			    all_samples = samples
			else:
			    all_samples = np.concatenate((all_samples, samples), axis=0)
			#print('all_samples', all_samples.shape)

		canvas = np.zeros_like(all_samples)
		for s in range(n_styles):
			for c in range(self.y_dim):
				canvas[s * self.y_dim + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

		save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], image_path=self.save_dir + '/result_' + str(epoch) + 'test_all_classes.png')


	def build_model(self):
		"""
		GAN model build and loss functions set up
		"""
		with tf.Graph().as_default():

			train_data, train_label = self.load_prepared_data()
			print train_data.shape
			print train_label.shape
			test_labels = train_label[0:self.batch_size]

			print "The shape of train_data is ", train_data.shape
			#train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.training_epoch)
			#iterator = train_dataset.make_one_shot_iterator()
			#one_element = iterator.get_next()

			noise_limit = 0.35
			L2_PENALTY = 0.02

			self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 28, 28, 1), name='real_images') # training examples
			self.y = tf.placeholder(tf.float32, shape=(self.batch_size, self.y_dim), name='y')
			self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 100), name='z') # Noises that feed to the generator 
			is_training = tf.placeholder(tf.bool)

			
			G_z = self.generator(self.z, self.y, is_training=is_training)
			D_real, D_real_logits, _ = self.discriminator(self.x, self.y, is_training=is_training) # D_real: the real convoluted training examples 
																				 	# D_real_logit: the convoluted but not activated training examples
			D_fake, D_fake_logits, _ = self.discriminator(G_z, self.y, is_training=is_training, reuse=True)

			D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
			D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

			D_loss_total = D_loss_real + D_loss_fake

			G_loss_total = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

			T_vars = tf.trainable_variables()
			D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
			G_vars = [var for var in T_vars if var.name.startswith('generator')]
			
			D_optimize = tf.train.AdamOptimizer(self.D_learning_rate, beta1=0.5).minimize(D_loss_total, var_list=D_vars)
			G_optimize = tf.train.AdamOptimizer(self.G_learning_rate, beta1=0.5).minimize(G_loss_total, var_list=G_vars)

			# For testing
			self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

			D_loss_real_sum = tf.summary.scalar("D_loss_real", D_loss_real)
			D_loss_fake_sum = tf.summary.scalar("D_loss_fake", D_loss_fake)
			D_loss_total_sum = tf.summary.scalar("D_loss_total", D_loss_total)
			G_loss_total_sum =  tf.summary.scalar("G_loss_total", G_loss_total)

			G_sum = tf.summary.merge([D_loss_fake_sum, G_loss_total_sum])
			D_sum = tf.summary.merge([D_loss_real_sum, D_loss_total_sum])
			#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
			sess = tf.Session()

			#--------------------------------Model-Training-------------------------------------------------------------
			sess.run(tf.global_variables_initializer())

			writer = tf.summary.FileWriter('/home/chenhui/fyp/logs/cgan', sess.graph)
			self.saver = tf.train.Saver()
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
					
					batch_images = train_data[j*self.batch_size:(j+1)*self.batch_size]
					batch_labels = train_label[j*self.batch_size:(j+1)*self.batch_size]

					#assert not np.any(np.isnan(batch_images)), "Error! The array becomes NaN"
					#print temp.shape

					z_ = np.random.normal(-1, 1, [self.batch_size, 100])
					loss_d, _, summary_str = sess.run([D_loss_total, D_optimize, D_sum], feed_dict={self.x: batch_images, self.y: batch_labels, self.z: z_, is_training: True})
					writer.add_summary(summary_str, count)
					D_losses.append(loss_d)

					count += 1
					#z_ = np.random.normal(0, 1, [self.batch_size, 100])
					loss_g, _, summary_str  = sess.run([G_loss_total, G_optimize, G_sum], feed_dict={self.x: batch_images, self.y: batch_labels, self.z: z_, is_training: True})
					writer.add_summary(summary_str, count)
					G_losses.append(loss_g)

					if np.mod(count, 300) == 0:
						test_image = sess.run(self.fake_images, feed_dict={self.z: fixed_z, self.y: test_labels, is_training: True})
						manifold_h = int(np.floor(np.sqrt(self.batch_size)))
						manifold_w = int(np.floor(np.sqrt(self.batch_size)))

						#temp_dir = self.save_dir + '/result_' + str(i) + '_' + str(j) + '.png'
						#self.save_image(test_image[self.batch_size - 1].reshape(28, 28, 1), temp_dir)
						save_images(test_image[:manifold_h*manifold_w, :, :, :], [manifold_h, manifold_w], self.save_dir + '/result_train_' + str(count) + '_' + str(j) + '.png')
				
				train_history['D_losses'].append(np.mean(D_losses))
				train_history['G_losses'].append(np.mean(G_losses))
				print "At epoch " + str(i) + " Current D_losses are ", str(np.mean(train_history.get('D_losses')))
				print "At epoch " + str(i) + " Current G_losses are ", str(np.mean(train_history.get('G_losses')))

				self.visualize_results(i, sess)
			#self.saver.save(sess, '/home/chenhui/fyp/logs/model/result.model', global_step=self.training_epoch) #save the god damned model
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
