import os
import random

import numpy as np
import tensorflow as tf
from losses import *
from networks import *
from utils import *

ms_coco_dir = ["/home/franky/Downloads/test2014/", "/home/franky/Downloads/test2015/", "/home/franky/Downloads/train2014/", "/home/franky/Downloads/val2014/"]
batch_size = 2
total_iterations = 100000


def read(train_dir):
    file_names = os.listdir(train_dir)
    random.shuffle(file_names)
    filename_queue = tf.train.string_input_producer(file_names, capacity=1000, num_epochs=None)
