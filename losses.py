import tensorflow as tf
import vgg16
import vgg19
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model


def pixel_loss(real, fake, norm="l2", weight=1, name="pixel_loss"):
    with tf.name_scope(name):
        if norm is "l2":
            loss = tf.losses.mean_squared_error(real, fake)
        else:
            loss = tf.losses.absolute_difference(real, fake)
        summary = tf.summary.scalar(name, loss)
        return weight * loss, summary


def feature_loss(real, fake, net="vgg16", layer="2_2", source="keras", norm="l2", weight=1, name="feature_loss"):
    with tf.name_scope(name):
        if source is "keras":
            base_model = VGG16(weights='imagenet', include_top=False) if net is "vgg16" else VGG19(
                weights='imagenet', include_top=False)
            model = Model(input=base_model.input, output=base_model.get_layer(
                "block{}_conv{}".format(layer[0], layer[-1])).output)
            real_feature = model.predict(preprocess_input(real))
            fake_feature = model.predict(preprocess_input(fake))
            if norm is "l1":
                loss = tf.losses.absolute_difference(real_feature, fake_feature)
            else:
                loss = tf.losses.mean_squared_error(real_feature, fake_feature)
        else:
            vgg = vgg16.Vgg16() if net is "vgg16" else vgg19.Vgg19()
            vgg.build(tf.concat([real, fake], axis=0))
            real_feature, fake_feature = tf.split(
                getattr(vgg, "conv{}_{}".format(layer[0], layer[-1])))
            if norm is "l1":
                loss = tf.losses.absolute_difference(real_feature, fake_feature)
            else:
                loss = tf.losses.mean_squared_error(real_feature, fake_feature)
        summary = tf.summary.scalar(name, loss)
        return weight * loss, summary


def disc_loss(disc_real, disc_fake, mode="least_square", weight=1, real=None, fake=None, lam=10, name="disc_loss"):
    with tf.name_scope(name):
        if mode is "least_square":
            loss = tf.reduc_mean(tf.square(disc_real - 1.0) + tf.square(disc_fake))
        elif mode is "wasserstein":
            loss = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
        elif mode is "improved_wasserstein":
            alpha = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
            differences = fake - real
            interpolates = real + (alpha * differences)
            gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
            loss += lam * gradient_penalty
        summary = tf.summary.scalar(name, loss)
        return weight * loss, summary


def adv_loss(disc_fake, mode="least_square", weight=1, name="adv_loss"):
    with tf.name_scope(name):
        if mode is "least_square":
            loss = tf.reduc_mean(tf.square(disc_fake - 1.0))
        elif mode is "wasserstein":
            loss = - tf.reduce_mean(disc_fake)
        summary = tf.summary.scalar(name, loss)
        return weight * loss, summary


def gen_loss(adv_loss, content_loss, weights=[1, 1], name="gen_loss"):
    with tf.name_scope(name):
        loss = weights[0] * adv_loss + weights[1] * content_loss
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
