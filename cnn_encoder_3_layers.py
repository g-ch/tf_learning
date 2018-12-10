import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random

input_dimension_xy = 64
input_dimension_z = 36

encoder_para = {
    "kernel1": 5,
    "stride_xy1": 2,
    "stride_z1": 3,
    "channel1": 32,
    "pool1": 2,
    "kernel2": 3,
    "stride_xy2": 2,
    "stride_z2": 3,
    "channel2": 64,
    "kernel3": 3,
    "stride_xy3": 2,
    "stride_z3": 2,
    "channel3": 128,
    "out_dia": 2048
}

def conv3d_relu(x, kernel_shape, bias_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def deconv3d(x, kernel_shape, output_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.conv3d_transpose(x, filter=weights, output_shape=output_shape, strides=strides)


def max_pool(x, kernel_shape, strides):
    return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights_relu", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_relu", [neurals_num], initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.matmul(x, weights) + biases)


def softmax(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights_soft", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_soft", [neurals_num], initializer=tf.constant_initializer(0.0))
    return tf.nn.softmax(tf.matmul(x, weights) + biases)


def encoder(x):
    k1 = encoder_para["kernel1"]
    sxy1 = encoder_para["stride_xy1"]
    sz1 = encoder_para["stride_z1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"]

    k2 = encoder_para["kernel2"]
    sxy2 = encoder_para["stride_xy2"]
    sz2 = encoder_para["stride_z2"]
    d2 = encoder_para["channel2"]

    k3 = encoder_para["kernel3"]
    sxy3 = encoder_para["stride_xy3"]
    sz3 = encoder_para["stride_z3"]
    d3 = encoder_para["channel3"]

    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(x, [k1, k1, k1, 1, d1], [d1], [1, sxy1, sxy1, sz1, 1])

        with tf.variable_scope("pool1"):
            max_pool1 = max_pool(conv1, [1, p1, p1, p1, 1], [1, p1, p1, p1, 1])

        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(max_pool1, [k2, k2, k2, d1, d2], [d2], [1, sxy2, sxy2, sz2, 1])

        with tf.variable_scope("conv3"):
            conv3 = conv3d_relu(conv2, [k3, k3, k3, d2, d3], [d3], [1, sxy3, sxy3, sz3, 1])
            return conv3


def decoder(x, batch_size):
    k1 = encoder_para["kernel1"]
    sxy1 = encoder_para["stride_xy1"]
    sz1 = encoder_para["stride_z1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"]

    k2 = encoder_para["kernel2"]
    sxy2 = encoder_para["stride_xy2"]
    sz2 = encoder_para["stride_z2"]
    d2 = encoder_para["channel2"]

    k3 = encoder_para["kernel3"]
    sxy3 = encoder_para["stride_xy3"]
    sz3 = encoder_para["stride_z3"]
    d3 = encoder_para["channel3"]

    size_1 = [batch_size, 64, 64, 36, d1]
    size_2 = [batch_size, 16, 16, 6, d2]
    size_3 = [batch_size, 8, 8, 2, d3]

    special_sxy1 = sxy1 * p1
    special_sz1 = sz1 * p1

    # Use conv to decrease kernel number. Use deconv to enlarge map

    with tf.variable_scope("decoder"):
        with tf.variable_scope("conv0"):  # Middle layer, change nothing
            conv0 = conv3d_relu(x, [k3, k3, k3, d3, d3], [d3], [1, 1, 1, 1, 1])
            print "conv0 ", conv0

        with tf.variable_scope("deconv0"):
            deconv0 = deconv3d(conv0, [sxy3, sxy3, sz3, d3, d3], output_shape=size_3, strides=[1, sxy3, sxy3, sz3, 1])
            print "deconv0", deconv0.get_shape()
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(deconv0, [k3, k3, k3, d3, d2], [d2], [1, 1, 1, 1, 1])

        with tf.variable_scope("deconv1"):
            deconv1 = deconv3d(conv1, [sxy2, sxy3, sz3, d2, d2], output_shape=size_2, strides=[1, sxy2, sxy2, sz2, 1])
            print "deconv1", deconv1.get_shape()
        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(deconv1, [k2, k2, k2, d2, d1], [d1], [1, 1, 1, 1, 1])

        with tf.variable_scope("deconv2"):  # special, consider pooling and stride in conv1
            deconv2 = deconv3d(conv2, [special_sxy1, special_sxy1, special_sz1, d1, d1], output_shape=size_1, strides=[1, special_sxy1, special_sxy1, special_sz1, 1])
            print "deconv2", deconv2.get_shape()
        with tf.variable_scope("conv3"):
            conv3 = conv3d_relu(deconv2, [k1, k1, k1, d1, 1], [1], [1, 1, 1, 1, 1])
            print "conv3", conv3.get_shape()
            return conv3


if __name__ == '__main__':
    '''Random data generation'''
    choice_value = [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0]  # from 0/7 to 7/7

    data_num = 5

    print "generating data... Approximate time: " + str(data_num * 2.8 / 60.0) + " minutes"
    dia_xy = input_dimension_xy
    dia_z = input_dimension_z
    data_mat = [[[[[choice_value[random.randint(0, 7)]] for k in range(dia_z)] for j in range(dia_xy)] for i in range(dia_xy)] for n in range(data_num)]

    print "Data generation is completed!"

    '''Training'''
    x_ = tf.placeholder("float", shape=[None, dia_xy, dia_xy, dia_z, 1])

    batch_size = 5
    learning_rate = 1e-4

    encode_vector = encoder(x_)
    decode_result = decoder(encode_vector, batch_size)

    loss = tf.reduce_mean(tf.square(x_ - decode_result))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print "Start training"
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        total_batch = int(data_num / batch_size)

        for epoch in range(100):
            for i in range(total_batch):
                sess.run(train_step, feed_dict={x_: data_mat})  # training
            print "epoch: " + str(epoch)
            print('loss=%s' % sess.run(loss, feed_dict={x_: data_mat}))
            # if epoch % 100 == 0:
            #     saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/' + str(epoch) + '_mnist.ckpt')


