# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import imageio
import random
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

def read_split_image(img): #split the images for comparison
	mat = misc.imread(img).astype(np.float)
	side = int(mat.shape[1]/2)
	print(side)

	image_A = mat[:, :side] #target image
	image_B = mat[:, side:] #source image

	return image_A, image_B


#---------------unit-test-----------------------------------------
#directory = '/home/linkwong/Zeroshot-GAN/model/image.png'
#img_A, img_B = read_split_image(directory)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_A.png', img_A)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_B.png', img_B)

def shift_and_resize_image(img, shift_x, shift_y, width, height): #image augmentation
	w, h, _ = img.shape
	print (w, h)
	enlarged = misc.imresize(img, [width, height])

	return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]

#--------------------unit-test------------------------------------
#directory = '/home/linkwong/Zeroshot-GAN/model/image.png'
#image = misc.imread(directory)
#multiplier = random.uniform(1.00, 1.20) #a number between 1 and 1.2
#print (multiplier)
#w, h, _ = image.shape
#nw = int(multiplier * w) + 1
#nh = int(multiplier * h) + 1 
#shift_x = int(np.ceil(np.random.uniform(0.01, nw - w))) #a number between 0.01 and nw-w
#shift_y = int(np.ceil(np.random.uniform(0.01, nh - h))) #a number between 0.01 and nh-h

#image_A, image_B = read_split_image(directory)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_A.png', image_A)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_B.png', image_B)
#image_A = shift_and_resize_image(image_A, shift_x, shift_y, nw, nh)
#image_B = shift_and_resize_image(image_B, shift_x, shift_y, nw, nh)

#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_AA.png', image_A)
#misc.imsave('/home/linkwong/Zeroshot-GAN/model/image_BB.png', image_B)
#------------------------------------------------------------------

def scale_back(images): #necessary? 
	return (images + 1.)/2

def merge(images, size): #images merging

	h, w = images.shape[1], images.shape[2] #height and channels, mistake?
	img = np.zeros((h * size[0], w * size[1], 3))
	print(img.shape)
	print("size 1 %d " % size[1])
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j * h:j * h + h, i * w:i * w+w, :] = image
	return img

def save_concat_images(imgs, img_path):
	concated = np.concatenate(imgs, axis=1)
	misc.imsave(img_path, concated)
