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
EvalHandle = namedtupe("EvalHandle", ["encoder", "generator", "target", "source", "embedding"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])
