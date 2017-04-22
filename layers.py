import numpy as np
import tensorflow as tf


def get_filter(shape, init_mode="truncated_normal", name="filter"):
    # with tf.variable_scope(name):
    if init_mode in ["truncated_normal", "random_normal", "random_uniform", "random_gamma", "random_possion", "zeros", "ones"]:
        return tf.Variable(getattr(tf, init_mode)(shape=shape), name=name)
    elif init_mode is "he_init":
        return tf.Variable(tf.random_normal(shape, stddev=np.sqrt(2.0 / shape[0] / shape[1] / (shape[2] + shape[3]) * 2)), name=name)
    elif init_mode is "xavier_init":
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(False))


def conv(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=2, padding="SAME", add_bias=True, name="conv"):
    with tf.variable_scope(name):
        in_channels = batch_input.get_shape().as_list()[-1]
        filter = get_filter(shape=[filter_size, filter_size, in_channels,
                                   out_channels], init_mode=filter_init_mode)
        strides = [1, stride, stride, 1]
        batch_output = tf.nn.conv2d(input=batch_input, filter=filter,
                                    strides=strides, padding=padding)
        if add_bias is True:
            bias = tf.Variable(tf.zeros(out_channels), name="bias")
            batch_output = tf.nn.bias_add(batch_output, bias)
        return batch_output


def deconv(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=2, padding="SAME", add_bias=True, name="deconv"):
    with tf.variable_scope(name):
        batch, height, width, in_channels = batch_input.get_shape().as_list()
        output_shape = [batch, height * 2, width * 2, out_channels]
        filter = get_filter(shape=[filter_size, filter_size, out_channels,
                                   in_channels], init_mode=filter_init_mode)
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


def dropout(x, keep_prob=0.5, name="dropout"):
    with tf.variable_scope(name):
        return tf.nn.dropout(x, keep_prob=keep_prob)


def batch_norm(x, name='batch_norm'):
    with tf.variable_scope(name):
        out_channels = x.shape.as_list()[-1]
        return tf.nn.fused_batch_norm(x,
                                      offset=tf.Variable(tf.zeros(out_channels), name='offset'),
                                      scale=tf.Variable(tf.ones(out_channels), name='scale'),
                                      name=name)[0]

def layer_norm(x, name="layer_norm"):
    return tf.contrib.layers.layer_norm(x, center=True, scale=True, trainable=True, scope=name)


def dense(batch_input, units, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name="dense"):
    with tf.variable_scope(name):
        return tf.layers.dense(inputs=batch_input, units=units, activation=activation, kernel_initializer=kernel_initializer)


def pixel_shuffler(x, block_size=2, name="ps"):
    with tf.variable_scope(name):
        return tf.depth_to_space(x, block_size=block_size)


def conv_block(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=2, padding="SAME", mode="cba", activation=leaky_relu, name="conv_block"):
    with tf.variable_scope(name):
        batch_output = conv(batch_input, out_channels, filter_size,
                            filter_init_mode, stride, padding)
        ops = {
            "a": activation,
            "b": batch_norm,
            "l": layer_norm,
            "p": pixel_shuffler,
        }
        for op in mode[1:]:
            batch_output = ops[op](batch_output)
        return batch_output


def deconv_block(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=2, padding="SAME", mode="dbda", activation=relu, keep_prob=0.5, name="deconv_block"):
    with tf.variable_scope(name):
        batch_output = deconv(batch_input, out_channels, filter_size,
                              filter_init_mode, stride, padding)
        ops = {
            "a": activation,
            "b": batch_norm,
            "l": layer_norm,
            "d": dropout,
        }
        for op in mode[1:]:
            batch_output = ops[op](batch_output)
        return batch_output


def res_block(batch_input, out_channels, filter_size=3, filter_init_mode="truncated_normal", stride=1, padding="SAME", activation=relu, name="res_block"):
    with tf.variable_scope(name):
        conv_1 = conv_block(batch_input, out_channels, filter_size,
                            filter_init_mode, stride, padding, mode="cba", activation=activation, name="conv_block_1")
        conv_2 = conv_block(conv_1, out_channels, filter_size,
                            filter_init_mode, stride, padding, mode="cb", name="conv_block_2")
        return tf.add(conv_2, batch_input)
