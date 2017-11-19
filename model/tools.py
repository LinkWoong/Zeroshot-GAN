# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import imageio
import scipy.misc as misc
import numpy as np
from cStringIO import StringIO


#make sure that the sequence is multiple of batches, necessary?
def padding_seq(seq, batch_size):

	seq_len = len(seq)
	if seq_len % batch_size == 0:
		return seq

	else:
		padding = batch_size - (seq_len % batch_size)
		seq.extend(seq[:padding])
		return seq

def bytes_to_file(bytes_image):

	return StringIO(bytes_image)

def read_split_image(img):
	mat = misc.imread(img).astype(np.float)
	side = int(mat.shape[1]/2)
	print(side)

	image_A = mat[:, :side] #target image
	image_B = mat[:, side:] #source image

	return image_A, image_B

#directory = '/home/linkwong/Zeroshot-GAN/model/image.png'
#img_A, img_B = read_split_image(directory)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_A.png', img_A)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_B.png', img_B)

def shift_and_resize_image(img, shift_x, shift_y, width, height):

	