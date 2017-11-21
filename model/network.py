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

	def discriminator(self, image, is_training, reuse=False): #from pix2pix model
		with tf.variable_scope("discriminator"):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			h0 = leakyrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
			h1 = leakyrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
											is_training, scope="d_bn_1"))
			h2 = leakyrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
											is_training, scope="d_bn_2"))
			h3 = leakyrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
											is_training, scope="d_bn_3"))
			fc1 = fc_layer(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")
			fc2 = fc_layer(tf.reshape(h3, [self.batch_size, -1]), self.embedding_num, scope="d_fc2")

			return tf.nn.sigmoid(fc1), fc1, fc2
	
	def build_model(self, is_training=True, inst_norm=False, no_target_source=False):
		real_data = tf.placeholder(tf.float32, [self.batch_size, self.input_width, self.input_width,
					self.input_filters + self.output_filters], name='real_A_and_B_images')
		embedding_ids = tf.placeholder(tf.int64, shape=None, name="embedding_ids")
		no_target_data = tf.placeholder(tf.float32, [self.batch_size, self.input_width,
													self.input_width, self.input_filters + self.output_filters],
													name='no_target_A_and_B_images')

		no_target_id = tf.placeholder(tf.int64, shape=None, name='no_target_id')

		#target images
		real_target = real_data[:, :, :, :self.input_filters]

		#source images
		real_source = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]

		embedding = init_embedding(self.embedding_num, self.embedding_dim)# (40, 1, 1, 128)
		fake_target, encode_real_source = self.generator(real_source, embedding, embedding_ids,
															is_training=is_training, inst_norm=inst_norm)
		real_source_target = tf.concat([real_source, real_target], 3)#concat for discrimination
		fake_source_target = tf.concat([real_source, fake_target], 3)#concat for discrimination

		#discriminated ones
		real_D, real_D_logits, real_category_logits = self.discriminator(real_source_target,
														is_training=is_training, reuse=False)
		fake_D, fake_D_logits, fake_category_logits = self.discriminator(fake_source_target,
														is_training=is_training, reuse=True)
		encode_fake_target = self.encoder(fake_target, is_training, reuse=True)[0]

		#losses set up

		#constant loss
		const_loss = (tf.reduce_mean(tf.square(encode_real_source - encode_fake_target))) * self.Lconst_penalty

		#category loss (introduced in Google's one-shot paper)
		true_labels = tf.reshape(tf.one_hot(indices=embedding_ids, depth=self.embedding_num),
											[self.batch_size, self.embedding_num])
		real_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_category_logits,
											labels=true_labels))
		fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_logits,
											labels=true_labels))
		category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

		#binary real/fake loss(for discriminator)

		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
											labels=tf.ones_like(real_D)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
											labels=tf.zeros_like(fake_D)))

		#L1 loss between real and generated images

		l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_target - real_target))

		#tv_loss
		tv_loss = (tf.nn.l2_loss(fake_target[:, 1:, :, :] - fake_target[:, :self.output_width - 1, :, :])/self.output_width
					+ tf.nn.l2_loss(fake_target[:, :, 1:, :] - fake_target[:, :, :self.output_width - 1, :])/self.output_width) * self.Ltv_penalty

		#fool

		cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
									labels=tf.ones_like(fake_D)))

		discriminator_loss = d_loss_real + d_loss_fake + category_loss / 2.0
		generator_loss = cheat_loss + l1_loss + self.Lcategory_penalty * fake_category_loss + const_loss + tv_loss

		if no_target_source: #used to prevent discriminator saturation

			no_target_source = no_target_data[:, :, :, self.input_filters:self.output_filters]
			no_target_target, encode_no_target_source = self.generator(no_target_source, embedding, no_target_id,
														is_training=is_training, inst_norm=inst_norm, reuse=True)
			no_target_labels = tf.reshape(tf.one_hot(indices=no_target_id, depth=self.embedding_num),
											shape=[self.batch_size, self.embedding_num])
			no_target_source_target = tf.concat([no_target_source, no_target_target], 3)
			no_target_D, no_target_D_logits, no_target_category_logits = self.discriminator(no_target_source_target,
																			is_training=is_training, reuse=True)
			encode_no_target_target = self.encoder(no_target_target, is_training, reuse=True)[0]
			no_target_const_loss = tf.reduce_mean(tf.square(encode_no_target_source - encode_no_target_target)) * self.Lconst_penalty
			no_target_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_category_logits,
																							labels=no_target_labels))
			d_loss_no_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_D_logits, labels=tf.zeros_like(no_target_D)))
			cheat_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_D_logits, labels=tf.ones_like(no_target_D)))

			discriminator_loss = d_loss_real + d_loss_fake + d_loss_no_target +(category_loss + no_target_category_loss) / 3.0
			generator_loss = cheat_loss / 2.0 + (l1_loss + self.Lcategory_penalty * fake_category_loss + no_target_category_loss)/2.0 + (const_loss + no_target_const_loss) / 2.0 + tv_loss


		#summary for compuation graph

		d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
		d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)
		category_loss_summary = tf.summary.scalar("category_loss", category_loss)
		cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss)
		l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
		fake_category_loss_summary = tf.summary.scalar("fake_category_loss", fake_category_loss)
		const_loss_summary = tf.summary.scalar("const_loss", const_loss)
		discriminator_loss_summary = tf.summary.scalar("discriminator_loss", discriminator_loss)
		generator_loss_summary = tf.summary.scalar("generator_loss", generator_loss)
		tv_loss_summary = tf.summary.scalar("tv_loss", tv_loss)

		d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
											category_loss_summary, discriminator_loss_summary])
		g_merged_summary = tf.summary.merge([cheat_loss_summary, l1_loss_summary,
											fake_category_loss_summary, const_loss_summary,
											generator_loss_summary, tv_loss_summary])


		#add handles
		input_handle = InputHandle(real_data=real_data, embedding_id=embedding_id,
									no_target_data=no_target_data, no_target_id=no_target_id)

		loss_handle = LossHandle(d_loss=discriminator_loss, g_loss=generator_loss, const_loss=const_loss,
								l1_loss=l1_loss, category_loss=category_loss, cheat_loss=cheat_loss, tv_loss=tv_loss)

		eval_handle = EvalHandle(encoder=encode_real_source, generator=fake_target, target=real_target,
								source=real_source, embedding=embedding)

		summary_handle = SummaryHandle(d_merged=d_merged_summary, g_merged=g_merged_summary)

		#set self."input_handle" = input_handle
		setattr(self, "input_handle", input_handle)
		setattr(self, "loss_handle", loss_handle)
		setattr(self, "eval_handle", eval_handle)
		setattr(self, "summary_handle", summary_handle)

	#model.sess = sess
	def register_session(self, sess):
		self.sess = sess

	def retrieve_trainable_vars(self, freeze_encoder=False):
		t_vars = tf.trainable_variables() #return trainable variables (trainable=True by default)

		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in g_vars if 'g_' in var.name]

		if freeze_encoder:
			print("Freeze encoder weights!!!!!!!!!!!")
			g_vars = [var for var in g_vars if not ("g_e" in var.name)]

		return g_vars, d_vars

	def retrieve_handles(self):
		input_handle = getattr(self, "input_handle")
		loss_handle = getattr(self, "loss_handle")
		eval_handle = getattr(self, "eval_handle")
		summary_handle = getattr(self, "summary_handle")

		return input_handle, loss_handle, eval_handle, summary_handle

	def get_model_id_and_dir(self): #get model id and directory
		model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
		model_dir = os.path.join(self.checkpoint_dir, model_id)
		return model_id, model_dir

	def checkpoint(self, saver, step): #store the checkpoints in order to restore to previous state

		model_name = "network.model"
		model_id, model_dir = get_model_id_and_dir()

		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

	def restore_model(self, saver, model_dir):

		checkpoint = tf.train.get_checkpoint_state(model_dir)

		if checkpoint:
			saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("restored model %s" % model_dir)

		else:
			print("Failed to restore the model %s " % model_dir)

	






		
dirr = '/home/linkwong/Zeroshot-GAN/model'
reader = tf.WholeFileReader()
directory = tf.train.string_input_producer(['/home/linkwong/Zeroshot-GAN/model/image.png'])
key, value = reader.read(directory)


image_tensor = tf.image.decode_png(value)
initialize = tf.global_variables_initializer()

generator_dim = 64
discriminator_dim = 64
output_width = 256
embedding_num = 40
embedding_dim = 128

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

	embedding = init_embedding(embedding_num, embedding_dim)
	print(embedding.shape)


	coord.request_stop()
	coord.join(threads)
