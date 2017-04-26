import os
import random
import warnings

import numpy as np

import tensorflow as tf
from losses import *
from networks import *
from utils import *

batch_size = 1
total_steps = 100000
train_dir = "/home/franky/Desktop/train/"
log_dir = "/home/franky/Desktop/log/"
log_steps = 100
learning_rate = 1e-4
vgg16model_path = "/home/franky/Downloads/vgg16-20160129.tfmodel"
gan_mode = "wasserstein"
warnings.filterwarnings("ignore")


def get_ckpt_path(train_mode, network_mode, batch_size, learning_rate):
    return train_mode + "-" + network_mode + "_bs" + batch_size + "_lr" + learning_rate + "/"


def run_trianing():

    gt_list, hi_list = get_train_file_lists(train_dir=train_dir)
    real_haze = tf.placeholder(tf.float32, [batch_size, 256, 256, 6], name="real_haze")
    real, haze = real_haze[:, :, :, :3], real_haze[:, :, :, 3:6]
    fake = generator(haze)
    real_fake = tf.concat([real, fake], axis=-1)
    disc_real_fake = discriminator(real)
    disc_real, disc_fake = tf.split(tf.squeeze(disc_real_fake), 2)

    # get losses
    pixel_loss, pixel_loss_summary = get_pixel_loss(real, fake, norm="l1", weight=1)
    feature_loss, feature_loss_summary = get_feature_loss(real, fake, norm="l1", weight=1)
    adv_loss, adv_loss_summary = get_adv_loss(disc_fake, mode=gan_mode, weight=1)
    gen_loss, gen_loss_summary = get_gen_loss(adv_loss, pixel_loss)
    disc_loss, disc_summary = get_disc_loss(disc_real, disc_fake, gan_mode, discriminator, batch_size, real, fake, lam=10)

    # get train_ops
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    gen_vars = [var for var in all_vars if "generator" in var.name]
    gen_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)
    disc_vars = [var for var in all_vars if "discriminator" in var.name]
    disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=disc_vars, colocate_gradients_with_ops=True)

    # pretrain and train
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        # saver = tf.train.Saver()
        sess.run(init_op)
        # sess.run(fake, feed_dict={real_haze: get_train_batch(gt_list, hi_list, 0, batch_size)})
        for step in range(100):
            feed_dict = {real_haze: get_train_batch(gt_list, hi_list, step, batch_size)}
            print(sess.run([pixel_loss], feed_dict))
            # _ = sess.run(gen_train_op, feed_dict)
            # for var in all_vars:
            # print(var)
            # print(sess.run(feature_loss, feed_dict={real_haze: get_train_batch(gt_list, gt_list, step, batch_size)}))
    # plt.subplot(1, 3, 1)
    # plt.imshow(a[0, :, :, 0])
    # plt.subplot(1,3,2)
    # plt.imshow(pretrain_batch[1].eval())
    # plt.subplot(1,3,3)
    # plt.imshow(pretrain_batch[0].eval()-pretrain_batch[1].eval())

    # summary_op = tf.summary.merge_all()
    #
    # ckpt_path = get_ckpt_path("pretrain", "generator_ae", str(batch_size), str(learning_rate))
    # val_path = get_ckpt_path("pre_val", "generator_ae", str(batch_size), str(learning_rate))
    # if not os.path.exists(log_dir + ckpt_path):
    #     os.mkdir(log_dir + ckpt_path)
    # if not os.path.exists(log_dir + val_path):
    #     os.mkdir(log_dir + val_path)
    # train_writer = tf.summary.FileWriter(log_dir + ckpt_path, sess.graph)
    # val_writer = tf.summary.FileWriter(log_dir + val_path, sess.graph)

        plt.show()


if __name__ == '__main__':
    run_trianing()
