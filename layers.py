import numpy as np
import tensorflow as tf


def get_filter(shape, init_mode="truncated_normal", name="filter"):
    with tf.variable_scope(name):
        if init_mode in ["truncated_normal", "random_normal", "random_uniform", "random_gamma", "random_possion", "zeros", "ones"]:
            return getattr(tf, init_mode)(shape=shape)
        elif init_mode is "he_init":
            return tf.random_normal(shape, stddev=np.sqrt(2.0 / shape[0] / shape[1] / (shape[2] + shape[3]) * 2), name="he_init")


def conv(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=2, padding="SAME", add_bias=False, name="conv"):
    with tf.variable_scope(name):
        in_channels = batch_input.get_shape().as_list()[-1]
        filter_shape = [filter_size, filter_size, in_channels, out_channels]
        filter = get_filter(shape=filter_shape, init_mode="truncated_normal")
        strides = [1, stride, stride, 1]
        batch_output = tf.nn.conv2d(input=batch_input, filter=filter,
                                    strides=strides, padding=padding)
        if add_bias is True:
            bias = tf.Variable(tf.zeros([out_channels]), name="bias")
            batch_output = tf.nn.bias_add(conv, bias)
        return batch_output


def deconv(batch_input, out_channels, filter_size=3, stride=2, padding="SAME", add_bias=False, name="deconv"):
    with tf.variable_scope(name):
        batch, height, width, in_channels = batch_input.get_shape()
        output_shape = [batch, height * 2, width * 2, out_channels]
        filter = get_filter(shape=[filter_size, filter_size, out_channels,
                                   in_channels], init_mode="truncated_normal")
        strides = [1, stride, stride, 1]
        batch_output = tf.nn.conv2d_transpose(
            value=batch_input, filter=filter, output_shape=output_shape, strides=strides, padding=padding)
        if add_bias is True:
            bias = tf.Variable(tf.zeros([out_channels]), name="bias")
            batch_output = tf.nn.bias_add(batch_output, bias)
        return batch_output


def leaky_relu(x, alpha=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, alpha * x)


def relu(x, name="relu"):
    with tf.variable_scope(name):
        return tf.nn.relu(x)


def batch_norm(input, name='batch_norm'):
    with tf.variable_scope(name):
        out_channels = input.shape.as_list()[3]
        return tf.nn.fused_batch_norm(input,
                                      offset=tf.Variable(tf.zeros(out_channels), name='offset'),
                                      scale=tf.Variable(tf.ones(out_channels), name='scale'),
                                      name=name)[0]


def conv_block(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=2, padding="SAME", mode="cba", activation="relu", name="conv_block"):
    with tf.variable_scope(name):
        batch_output = conv(batch_input, out_channels, filter_size,
                            filter_init_mode, stride, padding)
        if mode is "cba":
            batch_output = relu(batch_norm(batch_output)) if activation is "relu" else leaky_relu(
                batch_norm(batch_ouput))
        elif mode is "cab":
            batch_output = batch_norm(relu(batch_output)) if activation is "relu" else batch_norm(
                leaky_relu(batch_ouput))
        return batch_output
