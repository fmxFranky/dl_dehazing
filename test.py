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

    # data_dict = np.load("/home/franky/Desktop/NYU_ResNet-UpProj.npy", encoding='latin1').item()
    # for data in data_dict:
    #     print(data, end='\t')
    #     for d in data_dict[data]:
    #         print(d, end='  ')
    #     print()
    hazy_img = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], name="hazy_img")
    result = generator(hazy_img, mode="resnet", nb_res_blocks=8)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, "/home/franky/Desktop/save/lsgan_resnet8_bs4_lr5e-05/50000_step/model.ckpt")
        gt_list, hi_list = get_train_file_lists(train_dir=train_dir, num_epochs=1, shuffle=False)
        for step in range(10):
            train_feed_dict = {hazy_img: get_val_batch(hi_list, batch_size, step)}
            _result, _hazy_img = sess.run([result, hazy_img], feed_dict=train_feed_dict)
            plt.figure(step)
            plt.subplot(131)
            plt.imshow(np.uint8(_result[0]))
            plt.subplot(132)
            plt.imshow(io.imread("/home/franky/Desktop/train/{}_hazy.jpg".format(step + 1)))
            plt.subplot(133)
            plt.imshow(io.imread("/home/franky/Desktop/train/{}_ground_truth.jpg".format(step + 1)))

        plt.show()
    #         print(_feature_loss)
    # if step == 0:
    # a = io.imread("/home/franky/Desktop/a.jpg")
    # b = io.imread("/home/franky/Desktop/b.jpg")
    # print(measure.compare_mse(a,b))
    # plt.show()
