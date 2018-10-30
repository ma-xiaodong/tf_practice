import os
import time
from datetime import datetime

import tensorflow as tf
import mnist_inference

import pdb

def get_input():
    filename_queue = tf.train.string_input_producer([mnist_inference.DATA_PATH])
    reader = tf.TFRecordReader()
    pdb.set_trace()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features = {
	'image_raw': tf.FixedLenFeature([], tf.string),
	'pixels': tf.FixedLenFeature([], tf.int64),
	'label': tf.FixedLenFeature([], tf.int64)})


def main(argv = None):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        x, y_ = get_input()

if __name__ == '__main__':
    tf.app.run()

