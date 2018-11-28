import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random

input_side_diamension = 64

device = {
    "gpu1": "0",
    "gpu2": "0"
}

encoder_para = {
    "kernel1": 5,
    "stride1": 2,  # do not change
    "channel1": 32,
    "kernel2": 3,
    "stride2": 1,  # do not change
    "channel2": 32,
    "pool1": 4,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "pool2": 2,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "outdia": 2048
}

pooled_side_len1 = input_side_diamension
pooled_side_len2 = int(input_side_diamension / (encoder_para["pool1"] * encoder_para["stride1"]))

pooled_size = int(input_side_diamension * input_side_diamension * input_side_diamension * encoder_para["channel2"] / math.pow(encoder_para["pool1"]*encoder_para["pool2"]*encoder_para["stride1"], 3))


def conv3d_relu(x, kernel_shape, bias_shape, strides):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.0))
        with tf.device("/gpu:"+device["gpu2"]):
            conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
            return tf.nn.relu(conv + biases)


def deconv3d(x, kernel_shape, output_shape, strides):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        with tf.device("/gpu:" + device["gpu1"]):
            return tf.nn.conv3d_transpose(x, filter=weights, output_shape=output_shape, strides=strides)


def max_pool(x, kernel_shape, strides):
    with tf.device("/gpu:" + device["gpu1"]):
        return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu(x, x_diamension, neurals_num):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_relu", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias_relu", [neurals_num], initializer=tf.constant_initializer(0.0))
        with tf.device("/gpu:" + device["gpu1"]):
            return tf.nn.relu(tf.matmul(x, weights) + biases)


def softmax(x, x_diamension, neurals_num):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_soft", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias_soft", [neurals_num], initializer=tf.constant_initializer(0.0))
        with tf.device("/gpu:" + device["gpu1"]):
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

    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(x, [k1, k1, k1, 1, d1], [d1], [1, s1, s1, s1, 1])
            max_pool1 = max_pool(conv1, [1, p1, p1, p1, 1], [1, p1, p1, p1, 1])

        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(max_pool1, [k2, k2, k2, d1, d2], [d2], [1, s2, s2, s2, 1])
            max_pool2 = max_pool(conv2, [1, p2, p2, p2, 1], [1, p2, p2, p2, 1])
            print "max_pool2 ", max_pool2
            return max_pool2


def decoder(x, batch_size):
    k2 = encoder_para["kernel2"]
    s2 = encoder_para["stride2"]
    d2 = encoder_para["channel2"]
    p2 = encoder_para["pool2"]

    k1 = encoder_para["kernel1"]
    s1 = encoder_para["stride1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"] * s1

    pl1 = pooled_side_len1
    pl2 = pooled_side_len2

    with tf.variable_scope("decoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(x, [k2, k2, k2, d2, d2], [d2], [1, 1, 1, 1, 1])
            print "conv1 ", conv1

        with tf.variable_scope("deconv1"):
            deconv1 = deconv3d(conv1, [p2, p2, p2, d2, d2], output_shape=[batch_size, pl2, pl2, pl2, d2], strides=[1, p2, p2, p2, 1])
            print "deconv1", deconv1.get_shape()
        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(deconv1, [k2, k2, k2, d2, d1], [d1], [1, 1, 1, 1, 1])

        with tf.variable_scope("deconv2"):
            deconv2 = deconv3d(conv2, [p1, p1, p1, d1, d1], output_shape=[batch_size, pl1, pl1, pl1, d1], strides=[1, p1, p1, p1, 1])
            print "deconv2", deconv2.get_shape()
        with tf.variable_scope("conv3"):
            conv3 = conv3d_relu(deconv2, [k1, k1, k1, d1, 1], [1], [1, 1, 1, 1, 1])
            print "conv3", conv3.get_shape()
            return conv3


def generate_sin_x_plus_y(number, side_dim, step, start_x, start_y):
    """
    Generate 3d cube data to fit sin(x+y) + cos(z)= f(sin(x_t0), sin(y_t0)...)

    sin(x) and sin(y) for each slide

    out_put:
    data1: [number, side_dim, side_dim, side_dim, 1]
    data2: [number, z_dim]
    label: [number, 2]
    """
    x = start_x
    y = start_y

    data1 = []

    for i in range(number):
        sx = math.sin(x)
        sy = math.sin(y)

        # data1
        cube = []
        for j in range(side_dim):
            if j % 2 == 0:
                cube.append([[sx for m in range(side_dim)] for n in range(side_dim)])
            else:
                cube.append([[sy for m in range(side_dim)] for n in range(side_dim)])

        data1.append(cube)

        # update seed
        x = x + step
        y = y + step

    return np.array(data1)


if __name__ == '__main__':
    '''Random data generation'''
    # choice_value = [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0]  # from 0/7 to 7/7
    #
    data_num = 30
    #
    # print "generating data... Approximate time: " + str(data_num * 2.8 / 60.0) + " minutes"
    dia = input_side_diamension
    # data_mat = [[[[[choice_value[random.randint(0, 7)]] for k in range(dia)] for j in range(dia)] for i in range(dia)] for n in range(data_num)]
    #
    # print "Data generation is completed!"
    cube_dim = input_side_diamension

    data1 = generate_sin_x_plus_y(data_num, cube_dim, 0.1, 0, 0.5)
    data1 = data1.reshape(data_num, cube_dim, cube_dim, cube_dim, 1)

    '''Training'''
    x_ = tf.placeholder("float", shape=[None, dia, dia, dia, 1])

    batch_size = 30
    learning_rate = 1e-4

    encode_vector = encoder(x_)

    print "encode_vector: ", encode_vector.get_shape()
    decode_result = decoder(encode_vector, batch_size)

    loss = tf.reduce_mean(tf.square(x_ - decode_result))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print "Start training"
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        total_batch = int(data_num / batch_size)

        for epoch in range(2000):
            for i in range(total_batch):
                sess.run(train_step, feed_dict={x_: data1})  # training

            print "epoch: " + str(epoch)
            print('loss=%s' % sess.run(loss, feed_dict={x_: data1}))
            if epoch % 100 == 0:
                saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/' + str(epoch) + '_autoencoder.ckpt')


