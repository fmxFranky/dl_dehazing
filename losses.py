import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import vgg16
import vgg19
from utils import *

vgg16model_path = "/home/franky/Desktop/vgg16-20160129.tfmodel"


def get_pixel_loss(real, fake, norm="l2", weight=1, name="pixel_loss"):
    if norm is "l2":
        loss = tf.losses.mean_squared_error(real, fake)
    else:
        loss = tf.losses.absolute_difference(real, fake)
    loss *= weight
    # summary = tf.summary.scalar(name, loss)
    return loss, loss


def get_feature_loss(real, fake, model_file_type="npy", layer="4_2", norm="l2", weight=1, name="feature_loss"):
    if model_file_type is "npy":
        vgg_graph_def = tf.GraphDef()
        with open(vgg16model_path, "rb") as f:
            file_content = f.read()
        vgg_graph_def.ParseFromString(file_content)
        resized_images = tf.image.resize_area(tf.concat([real, fake], axis=0), [224, 224])
        tf.import_graph_def(vgg_graph_def, input_map={"images": resized_images})
        vgg_graph = tf.get_default_graph()
        feature_real_fake = vgg_graph.get_tensor_by_name("import/conv{}/Relu:0".format(layer))
        feature_real, feature_fake = tf.split(feature_real_fake, 2)
    elif model_file_type is "tfmodel":
        vgg = vgg16.Vgg16("/home/franky/Desktop/vgg16.npy")
        resized_batch = tf.concat([tf.image.resize_area(real, [224, 224]), tf.image.resize_area(fake, [224, 224])], axis=0)
        vgg.build(resized_batch)
        feature_real, feature_fake = tf.split(getattr(vgg, "conv{}".format(layer)), 2)
    if norm is "l2":
        loss = tf.losses.mean_squared_error(feature_real, feature_fake)
    else:
        loss = tf.losses.absolute_difference(feature_real, feature_fake)
    loss *= weight
    summary = tf.summary.scalar(name, loss)
    return loss, summary


def get_adv_loss(disc_fake, mode="least_square", weight=1, name="adv_loss"):
    if mode is "least_square":
        loss = tf.reduce_mean(tf.square(disc_fake - 1.0))
    else:
        loss = - tf.reduce_mean(disc_fake)
    loss *= weight
    summary = tf.summary.scalar(name, loss)
    return loss, summary


def get_gen_loss(adv_loss, content_loss, name="gen_loss"):
    with tf.name_scope(name):
        loss = adv_loss + content_loss
        summary = tf.summary.scalar(name, loss)
        return loss, summary


def get_disc_loss(disc_real, disc_fake, mode="least_square", discriminator=None, batch_size=None, real=None, fake=None, lam=10, name="disc_loss"):
    if mode is "least_square":
        loss = tf.reduc_mean(tf.square(disc_real - 1.0) + tf.square(disc_fake))
    elif mode is "wasserstein":
        loss = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
    elif mode is "improved_wasserstein":
        loss = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
        alpha = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        differences = fake - real
        interpolates = real + (alpha * differences)
        gradients = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        loss += lam * gradient_penalty
    summary = tf.summary.scalar(name, loss)
    return loss, summary


def batch_metric(real, fake):
    # real, fake are tensor.eval()
    mse = ((real - fake)**2).mean(axis=(1, 2))
    psnr = np.mean(20 * np.log10(255.0 / np.sqrt(mse)))
    average_real = real.mean((1, 2), keepdims=1)
    average_fake = fake.mean((1, 2), keepdims=1)
    stddev_real = real.std((1, 2), ddof=1)
    stddev_fake = fake.mean((1, 2), ddof=1)
    height, width = real.shape[1], real.shape[2]
    img_size = height * width
    covariance = ((real - average_real) * (average_fake)).mean((1, 2)) * img_size / (img_size - 1)
    average_real = np.squeeze(average_real)
    average_fake = np.squeeze(average_fake)
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * 255)**2, (k2 * 255)**2
    c3 = c2 / 2
    ssim = np.mean(((2 * average_real * average_fake + c1) * 2 * (covariance + c3)) / (average_fake**2 + average_real**2 + c1) / (stddev_fake**2 + stddev_real**2 + c2))
    return mse, psnr, ssim
