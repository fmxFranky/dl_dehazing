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
    from utils import *

    train_dir = "/home/franky/Desktop/train/"
    gt_list, hi_list = get_file_lists(train_dir)
    gt_batch, hi_batch = get_batch(gt_list, hi_list, 5)
    batch_size = 2
    total_steps = 100000
    train_dir = "/home/franky/Desktop/train/"
    log_dir = "/home/franky/Desktop/log/"
    log_steps = 100
    learning_rate = 1e-4
    # x = tf.random_uniform(minval=0, maxval=255, shape=[10, 256, 256, 3], name="img")
    # y = networks.discriminator(x, norm_mode="ln")
    # # z = networks.generator(x)
    # # b = tf.Variable(1)
    # y = conv(x, 64, filter_init_mode="xavier_init")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def get_ckpt_path(train_mode, network_mode, batch_size, learning_rate):
            return train_mode + "_" + network_mode + "_bs" + batch_size + "_lr" + learning_rate + "/"
        ckpt_path = get_ckpt_path("pretrain", "generator_ae", str(batch_size), str(learning_rate))
        if not os.path.exists(log_dir + ckpt_path):
            os.mkdir(log_dir + ckpt_path)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # i = 0
        # try:
        #     while not coord.should_stop() and i < 1:
        #         gt, hi = sess.run([gt_batch, hi_batch])
        #         for j in range(5):
        #             plt.subplot(5, 2, j * 2 + 1)
        #             plt.imshow(np.uint8(gt[j, :, :, :]))
        #             plt.subplot(5, 2, j * 2 + 2)
        #             plt.imshow(np.uint8(hi[j, :, :, :]))
        #         i += 1
        # except Exception as e:
        #     print("ok")
        # finally:
        #     coord.request_stop()
        # plt.show()
    #     # sess.run(tf.summary.merge_all())
        # vars = tf.trainable_variables()
        # for var in vars:
        #     print(var.name)
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
