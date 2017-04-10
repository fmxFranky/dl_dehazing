import tensorflow as tf
from layers import *


def generator(batch_input, mode="encoder_decoder", skip_connection=True, nb_res_blocks=16, train_mode=True,  name="generator"):
    with tf.variable_scope(name):
        if mode is "encoder_decoder":
            encoder_blocks = []
            encoder_channels = [64, 128, 256, 512, 512, 512, 512, 512]
            n = len(encoder_channels)
            encoder_blocks.append(conv_block(
                batch_input, encoder_channels[0], filter_size=4, mode="ca", activation=leaky_relu, name="encoder_block_1"))
            for e in range(n - 1):
                inputs = encoder_blocks[-1]
                encoder_blocks.append(conv_block(
                    inputs, encoder_channels[e + 1], filter_size=4, mode="cba", activation=leaky_relu, name="encoder_block_{}".format(e + 2)))
            decoder_blocks = []
            decoder_channels = [512, 512, 512, 512, 256, 128, 64, 3]
            decoder_blocks.append(deconv_block(
                encoder_blocks[-1], decoder_channels[0], filter_size=4, mode="dbda", activation=relu, name="decoder_block_1"))

            for d in range(n - 1):
                # print(encoder_blocks[n - d - 2], decoder_blocks[-1])
                inputs = tf.concat([encoder_blocks[n - d - 2], decoder_blocks[-1]],
                                   axis=3) if skip_connection is True else decoder_blocks[-1]
                if d < 2:
                    de_mode = "dbda"
                elif d < n - 2:
                    de_mode = "dba"
                else:
                    de_mode = "d"
                activation = relu if d < n - 2 else tf.nn.tanh
                decoder_blocks.append(deconv_block(
                    inputs, decoder_channels[d + 1], filter_size=4, mode=de_mode, activation=activation, name="decoder_block_{}".format(d + 2)))
            return decoder_blocks[-1]
        else:
            conv_1 = conv_block(batch_input, out_channels=64, filter_size=5, filter_init_mode="truncated_normal",
                                stride=1, mode="ca", activation=relu, name="conv_block_1")
            res_blocks = []
            for b in range(nb_res_blocks):
                res_blocks.append(res_block(
                    res_blocks[-1], out_channels=64, filter_size=3, stride=1, name="res_block_{}".format(b + 1)) if b else con_1)
            conv_2 = conv(res_blocks[-1], out_channels=64, filter_size=3, stride=1, name="conv_2")
            if train_mode:
                conv_2 = batch_norm(conv_2)
            conv_sum = tf.add(conv_2, conv_1)
            conv_3 = conv_block(conv_sum, out_channels=256, stride=1,
                                mode="cpa", activation=relu, name="conv_3")
            conv_4 = conv_block(conv_3, out_channels=256, stride=1,
                                mode="cpa", activation=relu, name="conv_4")
            batch_output = conv(conv_4, out_channels=3, stride=1, name="conv_5")
            return batch_output


def discriminator(batch_input, name="discriminator"):
    with tf.variable_scope(name):
        out_channels = [64, 128, 256, 512]
        conv_blocks = []
        for c in range(len(out_channels)):
            conv_blocks.append(conv_block(
                batch_input, out_channels[c], stride=1, mode="cab" if c else "ca", activation=leaky_relu, name="conv_{}_{}".format(c + 1, 1)))
            conv_blocks.append(conv_block(

                batch_input, out_channels[c], stride=2, mode="cab", activation=leaky_relu, name="conv_{}_{}".format(c + 1, 2)))
        dense_1 = dense(conv_blocks[-1], units=1024, activation=leaky_relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name="dense_1")
        dense_2 = dense(conv_blocks[-1], units=1, activation=tf.sigmoid,
                        kernel_initializer=tf.truncated_normal_initializer(), name="dense_2")
        return dense_2
