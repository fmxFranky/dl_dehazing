import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

if __name__ == '__main__':
    from layers import *
    import tensorflow as tf
    import networks
    import vgg16

    x = tf.random_uniform(minval=0, maxval=255, shape=[10, 256, 256, 3], name="img")
    y = networks.discriminator(x, norm_mode="ln")
    # # z = networks.generator(x)
    # # b = tf.Variable(1)
    # y = conv(x, 64, filter_init_mode="xavier_init")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    #     # sess.run(tf.summary.merge_all())
        vars = tf.trainable_variables()
        for var in vars:
            print(var.name)
    # x = io.imread("/home/franky/Desktop/vir_tf1.1_py3.5/projects/dl_dehazing/1_Depth_.bmp")
    # y = io.imread("/home/franky/Desktop/vir_tf1.1_py3.5/projects/dl_dehazing/1_Image_.bmp")
    # x = np.exp(-(np.array(x, dtype=np.float32) / 127))
    # print(x)
    # x = np.stack([x, x, x], axis=-1)
    # A = np.array([255, 255, 255], dtype=np.uint8)
    # y = np.array(y, dtype=np.float32)
    # z = np.array(y * x + A * (1 - x))
    # print(z)
    # plt.figure("1")
    # plt.imshow(np.uint8(z))
    # x = transform.resize(x, (256 * 2, 256 * 2), mode="reflect")
    # x = x * 255

    # print(x)
    # plt.imshow(x)
    # io.imsave("./dl_dehazing/aaa.jpg", np.uint8(x))
    # print("ss")
    # x = transform.resize(x, (256 * 2, 256 * 2))
    # io.imshow("b.jpg", x)
    # plt.show()
