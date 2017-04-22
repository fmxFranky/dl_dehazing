import os
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage import filters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def enhaze(J, A):
    height, width, channels = J.shape
    t = np.random.uniform(low=0.1, high=1, size=(height, width))
    t = filters.gaussian(t, sigma=10) + np.random.uniform(low=0.001,
                                                          high=0.01, size=(height, width))
    t = np.stack([t, t, t], axis=-1)
    # t =
    I = np.uint8(np.array(J).astype('float32') * t + A * (1 - t))
    io.imshow(I)
    plt.show()


def get_file_lists(train_dir):
    import tensorflow as tf
    ground_truths = [train_dir + "{}_Image_.jpg".format(i + 1) for i in range(1449)]
    hazy_images = [train_dir + "{}_Hazy.jpg".format(i + 1) for i in range(1449)]
    match = np.array([ground_truths, hazy_images]).transpose()
    np.random.shuffle(match)
    gt_list = match[:, 0]
    hi_list = match[:, 1]
    return gt_list, hi_list

    # img_names = os.listdir(coco_dir)
    # random.shuffle(img_names)
    # imgname_queue = tf.train.string_input_producer(img_names, )


def bmp2jpg(input_img_dir, output_img_dir):
    for img_name in os.listdir(input_img_dir):
        input_img = io.imread(input_img_dir + img_name)
        print(output_img_dir + img_name.split('.')[0] + '.jpg')
        io.imsave(output_img_dir + img_name.split('.')[0] + '.jpg', np.fliplr(input_img))


def get_batch(gt_list, hi_list, height, width, batch_size):
    tf_gt_list = tf.cast(gt_list, tf.string)
    tf_hi_list = tf.cast(hi_list, tf.string)
    input_queue = tf.train.slice_input_producer([tf_gt_list, tf_hi_list], num_epochs=200)
    gt_contents = tf.read_file(input_queue[0])
    hi_contents = tf.read_file(input_queue[1])
    ground_truths = tf.image.decode_jpeg(gt_contents)
    hazy_images = tf.image.decode_jpeg(hi_contents)
    ground_truths = tf.image.central_crop(ground_truths, 0.95)
    hazy_images = tf.image.central_crop(hazy_images, 0.95)


def main():
    # read("/home/franky/Desktop/train/")
    J = io.imread("/home/franky/Desktop/vir_tf1.1_py3.5/projects/dl_dehazing/demo_nyud_rgb.jpg")
    JJ = np.fliplr(J)
    io.imshow(JJ)
    plt.show()
    # A = [255, 255, 255]
    # enhaze(J, A)


if __name__ == '__main__':
    coco_dir = ["/home/franky/Downloads/test2014/", "/home/franky/Downloads/test2015/", "/home/franky/Downloads/train2014/", "/home/franky/Downloads/val2014/"]

    main()
