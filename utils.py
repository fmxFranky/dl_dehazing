import os
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage import util as sutil
from skimage import filters, transform
from skimage.morphology import disk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
IMG_HEIGHT = 256
IMG_WIDTH = 256
CROP_SIZE = 10


def enhaze_from_depth(j, d):
    J = transform.resize(j, [IMG_HEIGHT, IMG_HEIGHT], mode="reflect")
    beta = (1.5 - 0.5) * np.random.random_sample() + 0.5
    A = [(1.0 - 0.7) * np.random.random_sample() + 0.7] * 3
    t_ = np.exp(-beta * (np.array(d, dtype=np.float32) / 127.5))
    t = np.stack([t_, t_, t_], axis=-1)
    I = J * t + A * (1 - t)
    # [0,1]
    return I


def enhaze_from_random_transmission(j):
    # the value belongs to [0,1] for J,t,I
    t_ = filters.gaussian(0.9 * np.random.random_sample([IMG_HEIGHT, IMG_WIDTH]) + 0.1, sigma=10)
    t = np.stack([t_, t_, t_], axis=-1)
    J = transform.resize(j, [IMG_HEIGHT, IMG_HEIGHT], mode="reflect")
    A = [((1.0 - 0.7) * np.random.random_sample() + 0.7)] * 3
    I = J * t + A * (1 - t)
    return I


def crop_and_flip(img, scale="small"):
    h, w = np.shape(img)[0], np.shape(img)[1]
    if len(np.shape(img)) > 2:
        cropped_img = sutil.crop(img, ((CROP_SIZE, CROP_SIZE), (CROP_SIZE, CROP_SIZE), (0, 0)))
        if scale is "small":
            # img_scale is small ,r.g height, width<640
            lu = cropped_img[:256, :256, :]
            ru = cropped_img[:256, w - 256:w, :]
            ld = cropped_img[h - 256:h, :256, :]
            rd = cropped_img[h - 256:h, w - 256:w, :]
        else:
            print("wait to updates!")

    else:
        cropped_img = sutil.crop(img, ((10, 10), (10, 10)))
        if scale is "small":
            lu = cropped_img[:256, :256]
            ru = cropped_img[:256, w - 256:w]
            ld = cropped_img[h - 256:h, :256]
            rd = cropped_img[h - 256:h, w - 256:w]
        else:
            print("wait to updates!")
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


def get_train_file_lists(train_dir, num_epochs=1, shuffle=True):
    ground_truths = [train_dir + "{}_ground_truth.jpg".format(i) for i in np.arange(1449 * 8, dtype=np.int) + 1] * num_epochs
    hazy_images = [train_dir + "{}_hazy.jpg".format(i) for i in np.arange(1449 * 8, dtype=np.int) + 1] * num_epochs
    match = np.array([ground_truths, hazy_images]).transpose()
    if shuffle is True:
        np.random.shuffle(match)
    gt_list = match[:, 0]
    hi_list = match[:, 1]
    return gt_list, hi_list


def get_train_batch(gt_list, hi_list, batch_size, batch_idx):
    gt_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in gt_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    hi_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in hi_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    gt_batch = np.concatenate(gt_contents, axis=0)
    hi_batch = np.concatenate(hi_contents, axis=0)
    gt_hi_batch = np.concatenate([gt_batch, hi_batch], axis=-1)
    # the value belongs to [0,1] for batch
    return gt_hi_batch


def get_pretrain_file_list(pretrain_dir, shuffle=True):
    gt_list = [pretrain_dir + img for img in os.listdir(pretrain_dir)]
    return np.random.shuffle(gt_list) if shuffle else gt_list


def get_pretrain_batch(gt_list, batch_size, batch_idx):
    gt_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in gt_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    synth_contents = [np.expand_dims(enhaze_from_random_transmission(io.imread(img)), axis=0) for img in gt_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    gt_batch = np.concatenate(gt_contents, axis=0)
    synth_batch = np.concatenate(synth_contents, axis=0)
    gt_synth_batch = np.concatenate([gt_batch, synth_batch], axis=-1)
    return gt_synth_batch


def get_val_file_lists(val_dir, shuffle=True):
    val_ground_truths = [val_dir + "{}_ground_truth.jpg".format(i) for i in np.arange(1449 * 8, dtype=np.int) + 1]
    val_hazy_images = [val_dir + "{}_hazy.jpg".format(i) for i in np.arange(1449 * 8, dtype=np.int) + 1]
    match = np.array([val_ground_truths, val_hazy_images]).transpose()
    if shuffle is True:
        np.random.shuffle(match)
    val_gt_list = match[:, 0]
    val_hi_list = match[:, 1]
    return val_gt_list, val_hi_list


def get_val_batch(val_gt_list, val_hi_list, batch_size, select="ordered", batch_idx=None):
    if select is "ordered":
        assert start_idx is not None
        gt_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in val_gt_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
        hi_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in val_hi_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]]
    elif select is "random":
        selected_idxs = np.random.shuffle(range(len(val_gt_list)))[:batch_size]
        gt_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in [val_gt_list[idx] for idx in selected_idxs]]
        hi_contents = [np.expand_dims(transform.resize(io.imread(img), [IMG_HEIGHT, IMG_WIDTH], mode="reflect"), axis=0) for img in [val_hi_list[idx] for idx in selected_idxs]]
    gt_batch = np.concatenate(gt_contents, axis=0)
    hi_batch = np.concatenate(hi_contents, axis=0)
    gt_hi_batch = np.concatenate([gt_batch, hi_batch], axis=-1)
    return gt_hi_batch


def load_vgg16tfmodel(model_path, real, fake):
    import tensorflow as tf
    vgg_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        file_content = f.read()
    vgg_graph_def.ParseFromString(file_content)
    resize_images = tf.image.resize_area(tf.concat([real, fake], axis=0), [224, 224])
    tf.import_graph_def(vgg_graph_def, input_map={"images": resize_images})
    return


def log_train_information(step, pixel_loss, feature_loss, adv_loss, gen_loss, disc_loss, step_used_time, cur_total_time, mse, psnr, ssim, log_file, verbose=0):
    with open(log_file, "a+") as log:
        if step == 0:
            log.write("step\tpixel_loss\tfeature_loss\tadv_loss\tgen_loss\tdisc_loss\tstep_time_used\tcur_total_time\tmse\tpsnr\tssim\n")
        print("%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\n" % (step, pixel_loss, feature_loss, adv_loss, gen_loss, disc_loss, step_used_time, cur_total_time, mse, psnr, ssim))
    if verbose >= 0:
        print("step %d ~~ gen_loss: %.6f, disc_loss: %.6f; step_used_time: %.2fs; cur_total_time: %.2f\n" % (step, gen_loss, disc_loss, step_used_time, cur_total_time))
        if verbose == 1:
            print("         ~~ pixel_loss: %.6f; feature_loss: %.6f; adv_loss: %.6f; mse: %.6f; psnr: %.6f; ssim: %.6f\n" % (pixel_loss, feature_loss, adv_loss, mse, psnr, ssim))


def main():
    pass
    # plt.show()


if __name__ == '__main__':
    main()
