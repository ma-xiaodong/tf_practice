import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

mnist = input_data.read_data_set()
