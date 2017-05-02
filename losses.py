import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import vgg16
import vgg19
from utils import *

vgg16_tfmodel_path = "/home/franky/Desktop/vgg16-20160129.tfmodel"
vgg16_npy_path = "/home/franky/Desktop/vgg16.npy"


def get_pixel_loss(real, fake, norm="l2", weight=1, name="pixel_loss"):
    if norm is "l2":
        loss = tf.losses.mean_squared_error(real, fake)
        loss *= weight
        summary = tf.summary.scalar(name, loss)
        return loss, summary
    else:
        loss = tf.losses.absolute_difference(real, fake)
        loss *= weight
        summary = tf.summary.scalar(name, loss)
        return loss, loss


def get_feature_loss(real, fake, model_file_type="npy", layer="4_2", norm="l2", weight=1, name="feature_loss"):
    if model_file_type is "npy":
        vgg_graph_def = tf.GraphDef()
        with open(vgg16_tfmodel_path, "rb") as f:
            file_content = f.read()
        vgg_graph_def.ParseFromString(file_content)
        resized_images = tf.image.resize_images(tf.concat([real, fake], axis=0), [224, 224])
        tf.import_graph_def(vgg_graph_def, input_map={"images": resized_images})
        vgg_graph = tf.get_default_graph()
        feature_real_fake = vgg_graph.get_tensor_by_name("import/conv{}/Relu:0".format(layer))
        feature_real, feature_fake = tf.split(feature_real_fake, 2)
    elif model_file_type is "tfmodel":
        vgg = vgg16.Vgg16(vgg16_npy_path)
        resized_batch = tf.concat([tf.image.resize_images(real, [224, 224]), tf.image.resize_images(fake, [224, 224])], axis=0)
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


def get_gen_loss(adv_loss, content_loss, weight=1, name="gen_loss"):
    loss = adv_loss + content_loss
    loss *= weight
    summary = tf.summary.scalar(name, loss)
    return loss, summary


def get_disc_loss(disc_real, disc_fake, mode="improved_wgan", discriminator=None, batch_size=None, real=None, fake=None, lam=10, weight=1, name="disc_loss"):
    if mode is "lsgan":
        loss = tf.reduce_mean(tf.square(disc_real - 1.0) + tf.square(disc_fake))
    elif mode is "wgan":
        loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    elif mode is "improved_wgan":
        loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        alpha = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        differences = fake - real
        interpolates = real + (alpha * differences)
        gradients = tf.gradients(discriminator(interpolates, norm_mode="ln", reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        loss += lam * gradient_penalty
    loss *= weight
    summary = tf.summary.scalar(name, loss)
    return loss, summary


def batch_metric(real, fake):
    # real, fake are tensor.eval(), whose shape is [1,256,256,3]
    from skimage import measure
    assert real.shape == fake.shape
    if len(real.shape) == 3:
        mse = measure.compare_mse(real, fake)
        psnr = measure.compare_psnr(real, fake)
        ssim = measure.compare_ssim(real, fake, multichannel=True)
    else:
        mse = np.mean([measure.compare_mse(real[c], fake[c]) for c in range(real.shape[0])])
        psnr = np.mean([measure.compare_psnr(real[c], fake[c]) for c in range(real.shape[0])])
        ssim = np.mean([measure.compare_ssim(real[c], fake[c], multichannel=True) for c in range(real.shape[0])])
    return mse, psnr, ssim
