import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

if __name__ == '__main__':
    from layers import *
    import tensorflow as tf
    # import networks
    import vgg16

    x = tf.random_normal(shape=[10, 256, 256, 3], name="img")

    class ww(object):
        def __init__(self):
            self.conv2_2 = x
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = "conv2_2"
        q = ww()
        # y = networks.generator(x, skip_connection=True)
        print(getattr(q, a))
