import os
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage import util as sutil
from skimage import filters, transform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def enhaze_from_depth(j, d):
    beta = (1.5 - 0.5) * np.random.random_sample() + 0.5
    A = [np.uint8(((1.0 - 0.7) * np.random.random_sample() + 0.7) * 255)] * 3
    t = np.exp(-beta * (np.array(d, dtype=np.float32) / 127.5))
    t = np.stack([t, t, t], axis=-1)
    return np.uint8(np.array(j, dtype=np.float32) * t + A * (1 - t))


def crop_and_flip(img):
    h, w = np.shape(img)[0], np.shape(img)[1]
    if len(np.shape(img)) > 2:
        cropped_img = sutil.crop(img, ((10, 10), (10, 10), (0, 0)))
        lu = cropped_img[:256, :256, :]
        ru = cropped_img[:256, w - 256:w, :]
        ld = cropped_img[h - 256:h, :256, :]
        rd = cropped_img[h - 256:h, w - 256:w, :]
    else:
        cropped_img = sutil.crop(img, ((10, 10), (10, 10)))
        lu = cropped_img[:256, :256]
        ru = cropped_img[:256, w - 256:w]
        ld = cropped_img[h - 256:h, :256]
        rd = cropped_img[h - 256:h, w - 256:w]
    return lu, ru, ld, rd, np.fliplr(lu), np.fliplr(ru), np.fliplr(ld), np.fliplr(rd)


def bmp2jpg(gt_dir, hi_dir, output_img_dir):

    img_index = np.arange(1, 1449 * 8 + 1, dtype=np.int)
    np.random.shuffle(img_index)
    for i in range(1, 1449 + 1):
        print("preprocessing {}/1449 th image".format(i))
        gt = io.imread(gt_dir + "{}_Image_.bmp".format(i))
        dp = io.imread(gt_dir + "{}_Depth_.bmp".format(i))
        for (idx, part_gt, part_dp) in zip(range(8), crop_and_flip(gt), crop_and_flip(dp)):
            part_hazy = enhaze_from_depth(part_gt, part_dp)
            io.imsave(output_img_dir + "{}_ground_truth.jpg".format(img_index[(i - 1) * 8 + idx]), part_gt)
            io.imsave(output_img_dir + "{}_depth.jpg".format(img_index[(i - 1) * 8 + idx]), part_dp)
            io.imsave(output_img_dir + "{}_hazy.jpg".format(img_index[(i - 1) * 8 + idx]), part_hazy)


def get_train_file_lists(train_dir, num_epochs=1):
    import tensorflow as tf
    ground_truths = [train_dir + "{}_ground_truth.jpg".format(i) for i in np.arange(1449 * 8, dtype=np.int) + 1] * num_epochs
    hazy_images = [train_dir + "{}_hazy.jpg".format(i) for i in np.arange(1449 * 8, dtype=np.int) + 1] * num_epochs
    match = np.array([ground_truths, hazy_images]).transpose()
    # np.random.shuffle(match)
    gt_list = match[:, 0]
    hi_list = match[:, 1]
    return gt_list, hi_list


def get_train_batch(gt_list, hi_list, batch_idx, batch_size):
    import tensorflow as tf
    gt_contents = [np.expand_dims(transform.resize(io.imread(img), [256, 256], mode="reflect"), axis=0) for img in gt_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    hi_contents = [np.expand_dims(transform.resize(io.imread(img), [256, 256], mode="reflect"), axis=0) for img in hi_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    gt_batch = np.concatenate(gt_contents, axis=0)
    hi_batch = np.concatenate(hi_contents, axis=0)
    # print(np.mean(np.abs(gt_batch[0] - hi_batch[0])))
    real_haze_batch = np.concatenate([gt_batch, hi_batch], axis=-1)
    # the value belongs to [0,1]
    return real_haze_batch


def get_pretrain_file_list(pretrain_dir):
    import tensorflow as tf
    gt_list = [pretrain_dir + img for img in os.listdir(pretrain_dir)]
    return gt_list


def get_pretrain_batch(gt_list, batch_size):
    import tensorflow as tf
    tf_gt_list = tf.cast(gt_list, tf.string)
    transmission = tf.truncated_normal(shape=[batch_size, 256, 256, 1], mean=0.5, stddev=0.5)
    transmission = tf.concat([transmission, transmission, transmission], axis=-1)
    transmission = tf.image.resize_bicubic(transmission, size=[256, 256])
    return transmission


# def get_batch2
def load_vgg16(model_path, real, fake):
    import tensorflow as tf
    vgg_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        file_content = f.read()
    vgg_graph_def.ParseFromString(file_content)
    # with tf.get_variable_scope().reuse_variables():
    resize_images = tf.image.resize_area(tf.concat([real, fake], axis=0), [224, 224])
    print("ssss")
    tf.import_graph_def(vgg_graph_def, input_map={"images": resize_images})
    return


def main():
    pass
    # img = io.imread("/home/franky/Desktop/train/1_depth.jpg")
    # io.imshow(img)
    # plt.show(img.resize([256,256]))
    # pretrain_file_list = get_pretrain_file_list("/home/franky/Downloads/train2014/")[:10]
    # io.imsave("/home/franky/Desktop/1_depth.jpg",transform.resize(img,[256,256]))
    # get_file_lists(train_dir="/home/franky/Desktop/train/")
    # bmp2jpg("/home/franky/Desktop/D-HAZY_DATASET/NYU_GT/", "/home/franky/Desktop/D-HAZY_DATASET/NYU_Hazy/", "/home/franky/Desktop/train_dataset/")
    # J = io.imread("/home/franky/Desktop/vir_tf1.1_py3.5/projects/dl_dehazing/demo_nyud_rgb.jpg")
    # JJ = np.fliplr(J)
    # io.imshow("/home/franky/Desktop/train_dataset/2_depth.jpg")
    # plt.show()
    # A = [255, 255, 255]
    # enhaze(J, A)
    for i in os.listdir("/home/franky/Desktop/train/"):
        img = io.imread("/home/franky/Desktop/train/" + i)
        if img.shape[0] - 256 or img.shape[1] - 256:
            print(i)


if __name__ == '__main__':
    main()
