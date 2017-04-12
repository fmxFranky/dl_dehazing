import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

if __name__ == '__main__':
    from layers import *
    import tensorflow as tf
    import networks
    import vgg16

    x = tf.random_normal(shape=[10, 256, 256, 64], name="img")
    # y = networks.discriminator(x)
    # z = networks.generator(x)
    # b = tf.Variable(1)
    y = conv(x, 64, filter_init_mode="xavier_init")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.summary.merge_all())
        vars = tf.trainable_variables()
        for var in vars:
            print(var.name)
        # print(y)
