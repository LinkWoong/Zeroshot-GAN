# -*- coding: utf-8 -*-
import cPickle as pickle
import numpy as np
import random
import os
import tensorflow as tf
import scipy.misc as misc
from cStringIO import StringIO
from tensorflow.contrib.tensorboard.plugins import projector

def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq

def bytes_to_file(bytes_img):
    return StringIO(bytes_img)

def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B

def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    enlarged = misc.imresize(img, [nw, nh])
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]

def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized

class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples

def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h, _ = img_A.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            img_A = normalize_image(img_A)
            img_B = normalize_image(img_B)
            return np.concatenate([img_A, img_B], axis=2)
        finally:
            img.close()

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            labels = [e[0] for e in batch]
            processed = [process(e[1]) for e in batch]
            # stack into tensor
            yield labels, np.array(processed).astype(np.float32)

    return batch_iter()

class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)
        if self.filter_by:
            print("filter by label ->", filter_by)
            self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
            self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)
        print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
            for labels, examples in val_batch_iter:
                yield labels, examples

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path

    def get_all_examples(self):
        """
        Get all training images
        """
        train_length = len(self.train.examples)
        val_length = len(self.val.examples)

        return train_length, val_length

# This is 1 class, and the goal is not to treat it as 3370 classes
# On the contrast, the number of styles are treated as classes
# so if labels have size (3370) -> indicating 1 class (style)
# if labels have size (3370 *2 ) or something -> indicating 2 classes, each class has 3370 examples

data_dir = '/home/chenhui/zi2zi/experiment/data/char_zips/'
train_list = ['train_Font_Pair_No_1004.obj','train_Font_Pair_No_1005.obj',
                'train_Font_Pair_No_1006.obj','train_Font_Pair_No_1007.obj',
                'train_Font_Pair_No_1008.obj','train_Font_Pair_No_1009.obj']

val_list = ['val_Font_Pair_No_1004.obj','val_Font_Pair_No_1005.obj',
            'val_Font_Pair_No_1006.obj','val_Font_Pair_No_1007.obj',
            'val_Font_Pair_No_1008.obj','val_Font_Pair_No_1009.obj']

image_list = []
label_list = []

for i in range(0, 6):
    data_provider = TrainDataProvider(data_dir=data_dir, train_name=train_list[i], val_name=val_list[i])
    train_length, val_length = data_provider.get_all_examples()
    print "The length of training data is %d " % train_length
    print "The length of val data is %d " % val_length

    train_images = data_provider.get_train_iter(train_length)

    for index, batch in enumerate(train_images):
        labels, images = batch
        label_list.append(labels)
        image_list.append(images)
result = np.concatenate((image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5]), axis=0)
result = result[:,:,:3]
label_result = np.concatenate((label_list[0], label_list[1], label_list[2], label_list[3], label_list[4], label_list[5]), axis=0)
images = np.reshape(result, (20113, -1))
print images.shape # (20113, 24576)
#print "The length of labels is %d" % len(label_list)
print "The label shape is ", label_result.shape # (20113, )

#----------------projector implementation---------------

images_var = tf.Variable(images, name='images')
meta_data = os.path.join(data_dir, 'metadata.tsv')
with open(meta_data, 'w') as mf:
    for row in label_result:
        mf.write('%d\n' % row)

with tf.Session() as sess:
    saver = tf.train.Saver([images_var])
    print images_var.get_shape()
    sess.run(images_var.initializer)
    saver.save(sess, os.path.join(data_dir, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images_var.name
    embedding.metadata_path = meta_data
    projector.visualize_embeddings(tf.summary.FileWriter(data_dir), config)
