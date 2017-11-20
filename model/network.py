from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import scipy.misc as misc
import numpy as np 
import os
import time
from collections import namedtuple
from operators import *
from dataset import *
from tools import *

#Generative Adversarial Network (derived pix2pix,cGAN and acGAN) build
#It is complicated for me to understand these codes
#Hope no bug happens
#I will use handle instead of parser


#handles that will be used to feed into dict of tf.train
LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss", "const_loss", "l1_loss",
										"category_loss", "cheat_loss", "tv_loss"])
InputHandle = namedtuple("InputHandle", ["real_data", "embedding_id", "no_target_data", "no_target_id"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source", "embedding"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])


class network(object):
	def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256,
				output_width=256, generator_dim=64, discriminator_dim=64, L1_penalty=100,
				Lconst_penalty=15, Ltv_penalty=0.0, Lcategory_penalty=1.0, embedding_num=40,
				embedding_dim=128, input_filters=3, output_filters=3):
	#constructor built

		self.experiment_dir = experiment_dir
		self.experiment_id = experiment_id
		self.batch_size = batch_size
		self.input_width = input_width
		self.output_width = output_width
		self.generator_dim = generator_dim
		self.discriminator_dim = discriminator_dim
		self.L1_penalty = L1_penalty
		self.Lconst_penalty = Lconst_penalty
		self.Ltv_penalty = Ltv_penalty
		self.Lcategory_penalty = Lcategory_penalty
		self.embedding_num = embedding_num
		self.embedding_dim = embedding_dim
		self.input_filters = input_filters
		self.output_filters = output_filters
		self.sess = None  #??

		if experiment_dir: #store stuffs
			self.data_dir = os.path.join(self.experiment_dir, "data")
			self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
			self.sample_dir = os.path.join(self.experiment_dir, "sample")
			self.log_dir = os.path.join(self.experiment_dir, "log")

			if not os.path.exists(self.checkpoint_dir):
				os.makedirs(self.checkpoint_dir)
				print("create checkpoint directory")

			if not os.path.exists(self.sample_dir):
				os.makedirs(self.sample_dir)
				print("create sample directory")

			if not os.path.exists(self.log_dir):
				os.makedirs(self.log_dir)
				print("create log directory")

	def encoder(self, images, is_training, reuse=False): #return: (1, 1, 1, 512) and a dictionary
		with tf.variable_scope("generator"):
			if reuse:
				tf.get_variable_scope().reuse_variables()#reuse the variables

			encode_layers = dict()

			def encode_layer(x, output_filters, layer):
				act = leakyrelu(x)   #activate each batch
				conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
				enc = batch_norm(conv, is_training,  scope="g_e%d_bn" % layer) #norm each batch
				encode_layers["e%d" % layer] = enc
				return enc
			#images: [batch_size, 256, 256, 3]	
			e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
			encode_layers["e1"] = e1
			e2 = encode_layer(e1, self.generator_dim * 2, 2)
			e3 = encode_layer(e2, self.generator_dim * 4, 3)
			e4 = encode_layer(e3, self.generator_dim * 8, 4)
			e5 = encode_layer(e4, self.generator_dim * 8, 5)
			e6 = encode_layer(e5, self.generator_dim * 8, 6)
			e7 = encode_layer(e6, self.generator_dim * 8, 7)
			e8 = encode_layer(e7, self.generator_dim * 8, 8)

			return e8, encode_layers

	def decoder(self, encoded, encode_layers, id, inst_norm, is_training, reuse=False):
		with tf.variable_scope("generator"):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
				deconv = deconv2d(tf.nn.relu(x), [self.batch_size, output_width, output_width, output_filters],
							scope="g_d%d_deconv" % layer)

				if layer != 8:
					if inst_norm:
						print("Inst_norm")
					else:
						deconv = batch_norm(deconv, is_training, scope="g_d%d_deconv" % layer)
					if dropout:
						deconv = tf.nn.dropout(deconv, 0.5)
					if do_concat:
						deconv = tf.concat([deconv, enc_layer], 3)
				return deconv
			d1 = decode_layer(encoded, self.output_width/128, self.generator_dim * 8, layer=1, enc_layer=encode_layers["e7"],
								dropout=True)
			d2 = decode_layer(d1, self.output_width/64, self.generator_dim * 8, layer=2, enc_layer=encode_layers["e6"],
								dropout=True)
			d3 = decode_layer(d2, self.output_width/32, self.generator_dim * 8, layer=3, enc_layer=encode_layers["e5"],
								dropout=True)
			d4 = decode_layer(d3, self.output_width/16, self.generator_dim * 8, layer=4, enc_layer=encode_layers["e4"])
			d5 = decode_layer(d4, self.output_width/8, self.generator_dim * 4, layer=5, enc_layer=encode_layers["e3"])
			d6 = decode_layer(d5, self.output_width/4, self.generator_dim * 2, layer=6, enc_layer=encode_layers["e2"])
			d7 = decode_layer(d6, self.output_width/2, self.generator_dim, layer=7, enc_layer=encode_layers["e1"])
			d8 = decode_layer(d7, self.output_width, self.output_filters, layer=8, enc_layer=None, do_concat=False)
			print(type(d8))
			return tf.nn.tanh(d8) #scale (-1, 1)

	def generator(self, images, embeddings, embedding_ids, inst_norm, is_training, reuse=False):
		e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
		local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
		local_embeddings = tf.reshape(local_embeddings, [self.batch_size, 1, 1, self.embedding_dim]) #(16, 1, 1, 64)
		embedded = tf.concat([e8, local_embeddings], 3)
		output = self.decoder(embedded, enc_layers, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)

		return output, e8

	
dirr = '/home/linkwong/Zeroshot-GAN/model'
reader = tf.WholeFileReader()
directory = tf.train.string_input_producer(['/home/linkwong/Zeroshot-GAN/model/image.png'])
key, value = reader.read(directory)


image_tensor = tf.image.decode_png(value)
initialize = tf.global_variables_initializer()

generator_dim = 64
discriminator_dim = 64
output_width = 256

model = network(dirr, batch_size=1)

with tf.Session() as sess:

	sess.run(initialize)
	coord = tf.train.Coordinator()

	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(1):
		image = image_tensor.eval()

	image = tf.image.resize_images(image, [256, 256]) #resize the image into 256*256
	print(image.shape)

	image_ten = tf.convert_to_tensor(image, tf.float32) #convert the image into tensor
	image_ten = tf.expand_dims(image_ten, 0)
	print(image_ten.shape)

	e8, enc_layer = model.encoder(image_ten, is_training=True)
	print(type(e8))
	d8 = model.decoder(e8, enc_layer, 1, False, is_training=True)
	print(d8.shape) 
	coord.request_stop()
	coord.join(threads)
