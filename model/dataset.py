# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cPickle as pickle
import numpy as np 
import random
import os
import tools

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