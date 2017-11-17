import numpy as np
import tensorflow as tf
import scipy
import cPickle
import os 
import glob
import random
import imageio
import scipy.misc as misc


from __future__ import print_function
from __future__ import absolute_import

log_device_placement = True
allow_soft_placement = True

batch_size = 50
image_shape = [28*28]
z_dim = 30 #latent space reprsentation z proposed in the paper
gf_dim = 16
df_dim = 16
lr = 0.005
beta1 = 0.5


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):

	out = tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
										scale=True, is_training=is_training, scope=scope)
	return out

def conv2d(x, filter_size, stride_width, stride_height, feature_in, feature_out, scope="conv2d",log_device_placement=True):

	with tf.variable_scope(scope):

		w = tf.get_variable("w", [filter_size, filter_size, feature_in, feature_out], 
							initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable("b", [feature_out], initializer=tf.constant_initializer(0.0))
	conv = tf.nn.conv2d(x, w, strides=[1, stride_width, stride_height, 1], padding='SAME') + b

	return conv

def deconv2d(x, filter_size, stride_width, stride_height, feature_out, scope="deconv2d",log_device_placement=True):

	with tf.variable_scope(scope):

		w = tf.get_variable("w", [filter_size, filter_size, feature_out[-1], x.get_shape()[-1]], 
							initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable("b", [feature_out[-1]], initializer=tf.constant_intializer(0.0))

	deconv = tf.nn.conv2d_transpose(x, w, strides=[1, stride_width, stride_height, 1], output_shape=feature_out) + b

	return deconv

def leakyrelu(x, leak=0.2, name='lrelu'):

	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		out = f1 * x + f2 * abs(x)

	return out

def fc_layer(x, feature_in, feature_out, scope=None, with_w = False):

	with tf.variable_scope(scope or "Linear"):

		weights = tf.get_variable("weights", shape=[feature_in, feature_out], dtype=tf.float32, 
								initializer=tf.truncated_normal_initializer(stddev=0.02))

		bias = tf.get_variable("bias", shape=[feature_out], dtype=tf.float32, 
								initializer=tf.constant_initializer(0.0))
		if with_w:
			return tf.matmul(x, weights) + bias, weights, bias

		else:
			return tf.matmul(x, weights) + bias

def init_embedding(size, dimension, stddev=0.01, scope="Embedding"):

	with tf.variable_scope(scope):

		return tf.get_variable("E", shape=[size, 1, 1, dimension], dtype=tf.float32, 
			initializer=tf.truncated_normal_initializer(stddev=stddev))

def merge(image, size):

	height, width, channel = image[1], image[2], image[3]
	img = np.zeros(height * size[0], width * size[1], channel)
	print(img.shape)

	for i, j in enumerate(image):
		index = i % size[1]
		jndex = j / size[2]

		img[jndex*height:jndex*height + height, index*width:index*width + width] = image 
		#or img[jndex*height:jndex*height + height, index*width:index*width+width, :] = image

	return img

def image_norm(image):

	normalized = (image/127.5) - 1
	return image

def dense_batch_norm(x, number_out, phase_train, name='bn'): #BN necessary?

	beta = tf.get_variable(name + '/fc_beta', shape=[number_out], initializer=tf.constant_initializer(0.0))
	gamma = tf.get_variable(name + 'fc_gamma', shape=[number_out], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))

	batch_mean, batch_var = tf.nn.moments(x, [0], name=name + '/fc_moments')
	ema = tf.train.ExponentialMovingAverage(decay=0.9)

	def mean_var_update():

		ema_apply_op = ema.apply([batch_mean, batch_var])
		with tf.control_dependencies(ema_apply_op):
			return tf.identity(batch_mean), tf.identity(batch_var)
	mean ,var = tf.cond(name=phase_train, mean_var_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
	normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

	return normed

def global_batch_norm(x, number_out, phase_train, name='bn'): #BN necessary?

	beta = tf.get_variable(name + '/beta', shape=[number_out], initializer=tf.constant_initializer(0.0))
	gamma = tf.get_variable(name + '/gamma', shape=[number_out], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))

	batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name=name + '/moments')
	ema = tf.train.ExponentialMovingAverage(decay=0.9)

	def mean_var_update():

		ema_apply_op = ema.apply([batch_mean, batch_var])
		with tf.control_dependencies(ema_apply_op):
			return tf.identity(batch_mean), tf.identity(batch_var)

	mean, var = tf.cond(name=phase_train, mean_var_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
	normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

	return normed

def mini_batch_dis(x, num_kernels=100, dim_kernel=5, init=False, name='MD'):

	num_inputs = df_dim*4
	theta = tf.get_variable(name+'/theta', [num_inputs, num_kernels, dim_kernel], initializer=tf.random_normal_initializer(stddev=0.05))
