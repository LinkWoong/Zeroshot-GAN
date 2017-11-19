# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cPickle as pickle
import numpy as np 
import random
import os
import tools

#get each batch data during each iteration
def get_batch_iter(examples, batch_size, augment):

	padding = tools.padding_seq(examples, batch_size)

	def process(img): #enlarge the image and randomly crop to origin size
		img = tools.bytes_to_file(img)
		try:
			image_A, image_B = tools.read_split_image(img)
			if augment:
				w, h, _ = image_A.shape
				multiplier = random.uniform(1.00, 1.20)
				nw = int(multiplier * w) + 1
				nh = int(multiplier * h) + 1
				shift_x = int(np.ceil(np.random.uniform(0.01, nw-w)))
				shift_y = int(np.ceil(np.random.uniform(0.01, nh-h)))
				image_A = tools.shift_and_resize_image(image_A, shift_x, shift_y, nw, nh)
				image_B = tools.shift_and_resize_image(image_B, shift_x, shift_y, nw, nh)
			image_A = tools.normalize_image(image_A)
			image_B = tools.normalize_image(image_B)
			return np.concatenate([image_A, image_B], axis=2)

		finally:
			img.close()

	def batch_iter():
		for i in range(0, len(padding), batch_size):
			batch = padding[i:i + batch_size]
			labels = [e[0] for e in batch]
			processed = [process(e[1]) for e in batch] #do enlarge & crop
			yield labels, np.array(processed).astype(np.float32)

	return batch_iter()

#this class is used for reading and loading pickle files
class PickledImageProvider(object):
	def __init__(self, obj_path):
		self.obj_path = obj_path
		self.examples = self.load_pickled_examples()

	def load_pickled_examples(self):
		with open(self.obj_path, 'rb') as of:
			examples = list()

			while True:
				try:
					e = pickle.load(of)
					examples.append(e)
					if len(examples) % 100 == 0:
						print("processed %d examples" % len(examples))
				except EOFError:
					break
				except Exception:
					pass
				print("unpickled total %d examples" % len(examples))

				return examples

class TrainDataProvider(object): #read pickled training and validation data
	def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None):
		self.data_dir = data_dir
		self.filter_by = filter_by
		self.train_path = os.path.join(self.data_dir, train_name)
		self.val_path = os.path.join(self.data_dir, val_name)
		self.train = PickledImageProvider(self.train_path)
		self.val = PickledImageProvider(self.val_path)

		if self.filter_by:
			print("Filter by label ->", filter_by)
			self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
			self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)

		print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

	def get_train_iter(self, batch_size, shuffle=True): #get current traning images(enlarged)
		train_examples = self.train.examples[:]
		if shuffle:
			np.random.shuffle(train_examples)
		return get_batch_iter(train_examples, batch_size, augment=True)

	def get_val_iter(self, batch_size, shuffle=True):

		val_examples = self.val.examples[:]
		if shuffle:
			np.random.shuffle(val_examples)
		while True:
			val_batch_iter = get_batch_iter(val_examples, batch_size, augment=True)
			for labels, examples in val_batch_iter:
				yield labels, examples

	def compute_total_batch_num(self, batch_size): #get padded batch number
		return int(np.ceil(len(self.train.examples) / float(batch_size)))

	def get_all_labels(self):
		return list(e[0] for e in self.train.examples)

	def get_train_val_path(self):
		return self.train_path, self.val_path

class InjectDataProvider(object): #init embedding data provide
	def __init__(self, obj_path):
		self.data = TrainDataProvider(obj_path)
		print("examples -> %d" % len(self.data.examples))

	def get_single_embedding_iter(self, batch_size, embedding_id):
		exampels = self.data.examples
		batch_iter = get_batch_iter(examples, batch_size, augment=False)
		for _, images in batch_iter: # inject embedding style!!!!!asshole
			labels = [random.choice(embedding_id) for i in range(batch_size)]
			yield labels, images

class NeverEndLoopProvider(InjectDataProvider):#inherited from InjectDataProvider class, useful?
	def __init__(self, obj_path):
		super(NeverEndLoopProvider, self).__init__(obj_path)

	def get_random_embedding_iter(self, batch_size, embedding_id):
		while True:
			rand_iter = super(NeverEndLoopProvider, self).get_random_embedding_iter(batch_size, embedding_id)
			for labels, images in rand_iter:
				yield labels, images


