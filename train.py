import os
import random
import time
import warnings

import numpy as np
import tensorflow as tf

import vgg16
from losses import *
from networks import *
from utils import *

# network parameters
batch_size = 1
total_steps = 100000

train_dir = "/home/franky/Desktop/train/"
val_dir = "/home/franky/Desktop/val/"
log_steps = 100

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

# warnings.filterwarnings("ignore")


def get_ckpt_path(train_mode, network_mode, batch_size, learning_rate):
    return train_mode + "-" + network_mode + "_bs" + batch_size + "_lr" + learning_rate + "/"


def run_trianing():
    real_haze = tf.placeholder(tf.float32, [batch_size, 256, 256, 6], name="real_haze")
    real, haze = real_haze[:, :, :, :3], real_haze[:, :, :, 3:6]
    fake = generator(haze)
    real_fake = tf.concat([real, fake], axis=-1)
    disc_real_fake = discriminator(real_fake, norm_mode="ln" if gan_mode is "improved_wgan" else "bn", reuse=False)
    disc_real, disc_fake = tf.split(tf.squeeze(disc_real_fake), 2)
    # get losses and summaries
    pixel_loss, pixel_loss_summary = get_pixel_loss(real, fake, norm="l1", weight=100)
    feature_loss, feature_loss_summary = get_feature_loss(real, fake, model_file_type="tfmodel", norm="l1", weight=1)
    adv_loss, adv_loss_summary = get_adv_loss(disc_fake, mode=gan_mode, weight=1)
    gen_loss, gen_loss_summary = get_gen_loss(adv_loss, pixel_loss + feature_loss)
    disc_loss, disc_summary = get_disc_loss(disc_real, disc_fake, gan_mode, discriminator, batch_size, real, fake)

    # get train_ops
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    gen_vars = [var for var in all_vars if "generator" in var.name]
    disc_vars = [var for var in all_vars if "discriminator" in var.name]
    if gan_mode is "wgan":
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss, var_list=disc_vars, colocate_gradients_with_ops=True)
        clip_op = [var.assign(tf.clip_by_value(var, -weight_clip, weight_clip)) for var in disc_vars]
    else:
        gen_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=disc_vars, colocate_gradients_with_ops=True)

    summary_op = tf.summary.merge_all()

    # train
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init_op)
        gt_list, hi_list = get_train_file_lists(train_dir=train_dir, num_epochs=1)
        # val_gt_list, val_hi_list = get_val_file_lists(val_dir=val_dir, shuffle=True)
        train_statr_time = time.time()
        for step in range(100):
            step_start_time = time.time()
            train_feed_dict = {real_haze: get_train_batch(gt_list, hi_list, batch_size, step * disc_iters + np.random.randint(0, disc_iters))}
            _, _pixel_loss, _feature_loss, _adv_loss, _gen_loss = sess.run([gen_train_op, pixel_loss, feature_loss, adv_loss, gen_loss], feed_dict=train_feed_dict)
            for i in range(disc_iters):
                train_feed_dict = {real_haze: get_train_batch(gt_list, hi_list, batch_size, step * disc_iters + i)}
                _, _disc_loss = sess.run([disc_train_op, disc_loss], feed_dict=train_feed_dict)
            if gan_mode is "wgan":
                sess.run(clip_op)
            step_end_time = time.time()
            step_used_time = step_end_time - step_start_time
            cur_total_time = step_end_time - train_statr_time
            mse, psnr, ssim = batch_metric(real.eval(train_feed_dict), fake.eval(train_feed_dict))
            log_train_information(step, _pixel_loss, _feature_loss, _adv_loss, _gen_loss, _disc_loss, step_used_time, cur_total_time, mse, psnr, ssim, log_file=train_log_file, verbose=1)
            # if validation_mode is True:
            #     val_start_time = time.time()
            #     val_feed_dict = get_val_batch(val_gt_list, val_hi_list, batch_size, select="random")
            #     _pixel_loss, _feature_loss, _adv_loss, _gen_loss, _disc_loss = sess.run()
            # log_train_information(step, _pixel_loss, _feature_loss, _adv_loss, _gen_loss, _disc_loss, step_used_time, cur_total_time, mse, psnr, ssim, log_file=val_log_file, verbose=-1)

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
