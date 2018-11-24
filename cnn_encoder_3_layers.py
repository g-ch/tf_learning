import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random

input_side_diamension = 128

encoder_para = {
    "kernel1": 5,
    "stride1": 1,  # do not change
    "channel1": 32,
    "kernel2": 3,
    "stride2": 1,  # do not change
    "channel2": 64,
    "kernel3": 3,
    "stride3": 1,  # do not change
    "channel3": 64,
    "pool1": 4,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "pool2": 4,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "pool3": 2,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "outdia": 4096
}

pooled_side_len1 = input_side_diamension
pooled_side_len2 = int(input_side_diamension / (encoder_para["pool1"]))
pooled_side_len3 = int(input_side_diamension / (encoder_para["pool1"] * encoder_para["pool2"]))


pooled_size = int(input_side_diamension * input_side_diamension * input_side_diamension * encoder_para["channel2"] / math.pow(encoder_para["pool1"]*encoder_para["pool2"], 3))


def conv3d_relu(x, kernel_shape, bias_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def deconv3d(x, kernel_shape, output_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.conv3d_transpose(x, filter=weights, output_shape=output_shape, strides=strides)


def max_pool(x, kernel_shape, strides):
    return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights_relu", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_relu", [neurals_num], initializer=tf.constant_initializer(0.1))
    return tf.nn.relu(tf.matmul(x, weights) + biases)


def softmax(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights_soft", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_soft", [neurals_num], initializer=tf.constant_initializer(0.1))
    return tf.nn.softmax(tf.matmul(x, weights) + biases)


def encoder(x):
    k1 = encoder_para["kernel1"]
    s1 = encoder_para["stride1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"]

    k2 = encoder_para["kernel2"]
    s2 = encoder_para["stride2"]
    d2 = encoder_para["channel2"]
    p2 = encoder_para["pool2"]

    k3 = encoder_para["kernel3"]
    s3 = encoder_para["stride3"]
    d3 = encoder_para["channel3"]
    p3 = encoder_para["pool3"]

    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(x, [k1, k1, k1, 1, d1], [d1], [1, s1, s1, s1, 1])
            max_pool1 = max_pool(conv1, [1, p1, p1, p1, 1], [1, p1, p1, p1, 1])

        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(max_pool1, [k2, k2, k2, d1, d2], [d2], [1, s2, s2, s2, 1])
            max_pool2 = max_pool(conv2, [1, p2, p2, p2, 1], [1, p2, p2, p2, 1])

        with tf.variable_scope("conv3"):
            conv3 = conv3d_relu(max_pool2, [k3, k3, k3, d2, d3], [d3], [1, s3, s3, s3, 1])
            max_pool3 = max_pool(conv3, [1, p3, p3, p3, 1], [1, p3, p3, p3, 1])
            return max_pool3


def decoder(x, batch_size):
    k2 = encoder_para["kernel2"]
    s2 = encoder_para["stride2"]
    d2 = encoder_para["channel2"]
    p2 = encoder_para["pool2"]

    k1 = encoder_para["kernel1"]
    s1 = encoder_para["stride1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"]

    k3 = encoder_para["kernel3"]
    s3 = encoder_para["stride3"]
    d3 = encoder_para["channel3"]
    p3 = encoder_para["pool3"]

    pl1 = pooled_side_len1
    pl2 = pooled_side_len2
    pl3 = pooled_side_len3

    with tf.variable_scope("decoder"):
        with tf.variable_scope("conv0"):
            conv0 = conv3d_relu(x, [k3, k3, k3, d3, d3], [d3], [1, 1, 1, 1, 1])
            print "conv0 ", conv0

        with tf.variable_scope("deconv0"):
            deconv0 = deconv3d(conv0, [p3, p3, p3, d3, d3], output_shape=[batch_size, pl3, pl3, pl3, d3], strides=[1, p3, p3, p3, 1])
            print "deconv0", deconv0.get_shape()
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(deconv0, [k3, k3, k3, d3, d2], [d2], [1, s3, s3, s3, 1])

        with tf.variable_scope("deconv1"):
            deconv1 = deconv3d(conv1, [p2, p2, p2, d2, d2], output_shape=[batch_size, pl2, pl2, pl2, d2], strides=[1, p2, p2, p2, 1])
            print "deconv1", deconv1.get_shape()
        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(deconv1, [k2, k2, k2, d2, d1], [d1], [1, s2, s2, s2, 1])

        with tf.variable_scope("deconv2"):
            deconv2 = deconv3d(conv2, [p1, p1, p1, d1, d1], output_shape=[batch_size, pl1, pl1, pl1, d1], strides=[1, p1, p1, p1, 1])
            print "deconv2", deconv2.get_shape()
        with tf.variable_scope("conv3"):
            conv3 = conv3d_relu(deconv2, [k1, k1, k1, d1, 1], [1], [1, s1, s1, s1, 1])
            print "conv3", conv3.get_shape()
            return conv3


if __name__ == '__main__':
    '''Random data generation'''
    choice_value = [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0]  # from 0/7 to 7/7

    data_num = 5

    print "generating data... Approximate time: " + str(data_num * 2.8 / 60.0) + " minutes"
    dia = input_side_diamension
    data_mat = [[[[[choice_value[random.randint(0, 7)]] for k in range(dia)] for j in range(dia)] for i in range(dia)] for n in range(data_num)]

    print "Data generation is completed!"

    '''Training'''
    x_ = tf.placeholder("float", shape=[None, 128, 128, 128, 1])

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
            if epoch % 10 == 0:
                saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/' + str(epoch) + '_mnist.ckpt')


