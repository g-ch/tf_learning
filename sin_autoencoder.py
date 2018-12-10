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


def generate_sin_x_plus_y(number, side_dim_xy, side_dim_z, step, start_x, start_y):
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
        for j in range(side_dim_xy):
            if j < side_dim_xy / 2:
                cube.append([[sx for m in range(side_dim_z)] for n in range(side_dim_xy)])
            else:
                cube.append([[sy for m in range(side_dim_z)] for n in range(side_dim_xy)])

        data1.append(cube)

        # update seed
        x = x + step
        y = y + step

    return np.array(data1)


if __name__ == '__main__':
    '''Random data generation'''
    data_num = 30

    data1 = generate_sin_x_plus_y(data_num, input_dimension_xy, input_dimension_z, 0.1, 0, 0.5)
    data1 = data1.reshape(data_num, input_dimension_xy, input_dimension_xy, input_dimension_z, 1)

    '''Training'''
    x_ = tf.placeholder("float", shape=[None, input_dimension_xy, input_dimension_xy, input_dimension_z, 1])

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

        for epoch in range(100):
            for i in range(total_batch):
                sess.run(train_step, feed_dict={x_: data1})  # training

            print "epoch: " + str(epoch)
            print('loss=%s' % sess.run(loss, feed_dict={x_: data1}))
            if epoch % 20 == 0:
                saver.save(sess, '/home/clarence/log/model/' + str(epoch) + '_autoencoder.ckpt')


