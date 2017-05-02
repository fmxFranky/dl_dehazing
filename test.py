import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
batch_size = 1
total_steps = 100000

train_dir = "/home/franky/Desktop/train/"
val_dir = "/home/franky/Desktop/val/"
log_steps = 500

log_train_dir = "/home/franky/Desktop/log/train/"
log_val_dir = "/home/franky/Desktop/log/val/"
train_log_file = "/home/franky/Desktop/log/train_information.txt"
val_log_file = "/home/franky/Desktop/log/val_information.txt"
save_dir = "/home/franky/Desktop/save/"
vgg16model_path = "/home/franky/Desktop/vgg16-20160129.tfmodel"
gan_mode = "improved_wgan"
weight_clip = 0.01
disc_iters = 1 if gan_mode is "lsgan" else 5
learning_rate = 1e-4 if gan_mode is "wgan" else 5e-5
validation_mode = False
if __name__ == '__main__':
    from layers import *
    import tensorflow as tf
    import vgg16
    from utils import *
    from losses import *
    from networks import *
    from skimage import measure
    import pandas as pd

    # real_haze = tf.placeholder(tf.float32, [batch_size, 256, 256, 6], name="real_haze")
    # real, haze = real_haze[:, :, :, :3], real_haze[:, :, :, 3:6]
    # fake = generator(real)
    # real_synth = tf.placeholder(tf.float32, [batch_size, 256, 256, 6], name="real_haze")
    # real, synth = real_synth[:, :, :, :3], real_synth[:, :, :, 3:6]
    # feature_loss, feature_loss_summary = get_feature_loss(synth, haze, model_file_type="tfmodel", norm="l1", weight=1)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     gt_list, hi_list = get_train_file_lists(train_dir=train_dir, num_epochs=1, shuffle=False)
    #     synth_list = get_pretrain_file_list( / home/franky/Downloads/)
    #     for step in range(1):
    #         train_feed_dict = {real_haze: get_train_batch(gt_list, hi_list, batch_size, step), real_synth: get_pretrain_batch(gt_list, batch_size, step)}
    #         _haze, _synth, _feature_loss = sess.run([haze, synth, feature_loss], feed_dict=train_feed_dict)
    #         print(_feature_loss)
    # if step == 0:
    # plt.subplot(121)
    # plt.imshow(enhaze_from_random_transmission(_haze[0]))
    # plt.subplot(122)
    # plt.imshow(_synth[0])
    # plt.show()
