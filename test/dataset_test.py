# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cPickle as pickle
import numpy as np
import random
import os

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

data_dir = '/home/chenhui/zi2zi/experiment/'
label_dir = '/home/chenhui/zi2zi/experiment/'

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


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_single_embedding_iter(self, batch_size, embedding_id):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * batch_size
            yield labels, images

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [random.choice(embedding_ids) for i in range(batch_size)]
            yield labels, images

data_provider = TrainDataProvider(data_dir)
batch_size = 16
total_batch_num = data_provider.compute_total_batch_num(batch_size)
print("Total batch number is %d " % total_batch_num)
train_batch_iter = data_provider.get_train_iter(batch_size)
'''
for bid, batch in enumerate(train_batch_iter):
    labels, batch_images = batch
    print(labels)
    print(batch_images.shape)
'''
all_labels = data_provider.get_all_labels()
with open(os.path.join(label_dir, 'metadata.tsv'), 'w') as meta_file:
    for row in all_labels:
        meta_file.write('%d\n' % row)
