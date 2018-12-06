import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
test_data_num = 400
input_side_dimension = 64

states_num_one_line = 13
labels_num_one_line = 4

clouds_filename = "/home/ubuntu/chg_workspace/data/csvs/chg_route1_trial1/pcl_data_2018_12_03_11:34:18.csv"
states_filename = "/home/ubuntu/chg_workspace/data/csvs/chg_route1_trial1/uav_data_2018_12_03_11:34:18.csv"
labels_filename = "/home/ubuntu/chg_workspace/data/csvs/chg_route1_trial1/label_data_2018_12_03_11:34:18.csv"

img_wid = input_side_dimension
img_height = input_side_dimension

''' Parameters for RNN'''
rnn_paras = {
    "time_step": 5,
    "state_len": 256,
    "input_len": 2500,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 2048,  # should be the same as encoder out dim
    "dim2": 452  # dim1 + dim2 should be input_len of the rnn, for line vector
}

''' Parameters for CNN encoder'''
encoder_para = {
    "kernel1": 5,
    "stride1": 2,  # do not change
    "channel1": 32,
    "kernel2": 3,
    "stride2": 1,  # do not change
    "channel2": 32,
    "pool1": 4,  # 2 or 4 recommended, carefully choose according to input_side_dimmension to keep values interger
    "pool2": 2,  # 2 or 4 recommended, carefully choose according to input_side_dimmension to keep values interger
    "outdim": 2048
}

pooled_side_len1 = input_side_dimension
pooled_side_len2 = int(input_side_dimension / (encoder_para["pool1"] * encoder_para["stride1"]))

pooled_size = int(input_side_dimension * input_side_dimension * input_side_dimension * encoder_para["channel2"] / math.pow(encoder_para["pool1"]*encoder_para["pool2"]*encoder_para["stride1"], 3))


def myrnn_test(x, state_last, input_len, output_len, state_len):
    """
    RNN function
    x: [raw_batch_size, time_step, input_len]
    state dimension is also weights dimension in hidden layer
    output_len can be given as you want(same as label dimension)
    """
    with tf.variable_scope("rnn"):
        w = tf.get_variable("weight_x", [input_len, state_len])
        u = tf.get_variable("weight_s", [state_len, state_len])
        v = tf.get_variable("weight_y", [state_len, output_len])
        b = tf.get_variable("bias", [output_len])
        # state = tf.get_variable("state", [1, state_len], trainable=False, initializer=tf.constant_initializer(0.0))
        # state = tf.zeros([1, state_len])
        state = tf.nn.tanh(tf.matmul(state_last, u) + tf.matmul(x, w))  # hidden layer activate function
        return tf.nn.tanh(tf.matmul(state, v) + b), state  # output layer activate function


def conv3d_relu(x, kernel_shape, bias_shape, strides):
    """ 3D convolution For 3D CNN encoder """
    weights = tf.get_variable("weights_con", kernel_shape)  # truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_con", bias_shape)
    conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def max_pool(x, kernel_shape, strides):
    """ 3D convolution For 3D CNN encoder """
    return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def encoder(x):
    """
    3D CNN encoder function
    x: [raw_batch_size, input_len_x, input_len_y, input_len_z, 1]
    """

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
            # print "max_pool2 ", max_pool2
            return max_pool2


def read_pcl(data, filename):
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
            with open(filename, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_wid):
                        for j in range(img_wid):
                            for k in range(img_height):
                                data[i_row, i, j, k, 0] = row[i * img_wid + j * img_wid + k * img_height]
                    i_row = i_row + 1
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    return True


def read_others(data, filename, num_one_line):
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
            with open(filename, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(num_one_line):
                        data[i_row, i] = row[i]
                    i_row = i_row + 1
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    return True


class Networkerror(RuntimeError):
    """
    Error print
    """
    def __init__(self, arg):
        self.args = arg


if __name__ == '__main__':
    ''' Make some training values '''
    print "Reading data..."
    # Read clouds
    clouds = open(clouds_filename, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_mat = np.ones([img_num, img_wid, img_wid, img_height, 1])
    read_pcl(data_mat, clouds_filename)

    # Read states
    states = open(states_filename, "r")
    states_num = len(states.readlines())
    states.close()
    if states_num != img_num:
        raise Networkerror("states file mismatch!")
    states_mat = np.zeros([states_num, states_num_one_line])
    read_others(states_mat, states_filename, states_num_one_line)

    # Read labels
    labels = open(labels_filename, "r")
    labels_num = len(labels.readlines())
    labels.close()
    if labels_num != states_num:
        raise Networkerror("labels file mismatch!")
    labels_mat = np.zeros([labels_num, labels_num_one_line])
    read_others(labels_mat, labels_filename, labels_num_one_line)

    ''' Choose useful states and labels '''
    # compose_num = [256]
    # # check total number
    # num_total = 0
    # for num_x in compose_num:
    #     num_total = num_total + num_x
    # if num_total != concat_paras["dim2"]:
    #     raise Networkerror("compose_num does not match concat_paras!")
    # # concat for input2
    # states_input = np.concatenate([np.reshape(states_mat[:, 10], [states_num, 1]) for i in range(compose_num[0])],
    #                               axis=1)  # delt_yaw

    compose_num = [100, 100, 240, 6, 6]
    # check total number
    num_total = 0
    for num_x in compose_num:
        num_total = num_total + num_x
    if num_total != concat_paras["dim2"]:
        raise Networkerror("compose_num does not match concat_paras!")
    # concat for input2
    states_input_current_yaw_x = np.concatenate(
        [np.reshape(states_mat[:, 10], [states_num, 1]) for i in range(compose_num[0])], axis=1)  # current_yaw_x
    states_input_current_yaw_y = np.concatenate(
        [np.reshape(states_mat[:, 11], [states_num, 1]) for i in range(compose_num[1])], axis=1)  # current_yaw_y
    states_input_delt_yaw = np.concatenate(
        [np.reshape(states_mat[:, 12], [states_num, 1]) for i in range(compose_num[2])], axis=1)  # delt_yaw
    states_input_linear_vel = np.concatenate(
        [np.reshape(states_mat[:, 2], [states_num, 1]) for i in range(compose_num[3])], axis=1)  # linear vel
    states_input_angular_vel = np.concatenate(
        [np.reshape(states_mat[:, 3], [states_num, 1]) for i in range(compose_num[4])], axis=1)  # angular vel

    states_input = np.concatenate(
        [states_input_current_yaw_x, states_input_current_yaw_y, states_input_delt_yaw, states_input_linear_vel, states_input_angular_vel], axis=1)

    labels_ref = labels_mat[:, 0:2]  # vel_cmd, angular_cmd

    print "Data reading is completed!"

    ''' Graph building '''
    cube_data = tf.placeholder("float", name="cube_data", shape=[1, input_side_dimension, input_side_dimension, input_side_dimension, 1])
    line_data = tf.placeholder("float", name="line_data", shape=[1, concat_paras["dim2"]])
    state_data = tf.placeholder("float", name="line_data", shape=[1, rnn_paras["state_len"]])

    # 3D CNN
    encode_vector = encoder(cube_data)
    # To flat vector
    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["outdim"]])
    # Concat, Note: dimension parameter should be 1, considering batch size
    concat_vector = tf.concat([encode_vector_flat, line_data], 1)
    # Dropout
    # concat_vector = tf.layers.dropout(concat_vector, rate=0.3, training=True)
    # Feed to rnn
    result, state_returned = myrnn_test(concat_vector, state_data, rnn_paras["input_len"], rnn_paras["output_len"], rnn_paras["state_len"])

    ''' Predicting '''
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
    restorer = tf.train.Saver(variables_to_restore)

    cube_dim = input_side_dimension
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, "/home/ubuntu/chg_workspace/3dcnn/model/model_cnn_rnn_timestep5/simulation_cnn_rnn1600.ckpt")
        state_data_give = np.zeros([1, rnn_paras["state_len"]])

        results_to_draw = []
        for i in range(test_data_num):
            data1_to_feed = data_mat[i, :].reshape([1, cube_dim, cube_dim, cube_dim, 1])
            data2_to_feed = states_input[i, :].reshape([1, concat_paras["dim2"]])

            results = sess.run(result, feed_dict={cube_data: data1_to_feed, line_data: data2_to_feed, state_data: state_data_give})
            state_data_give = sess.run(state_returned, feed_dict={cube_data: data1_to_feed, line_data: data2_to_feed, state_data: state_data_give})

            results_to_draw.append(results)
            print "result: ", results, "label: ", labels_ref[i]

        results_to_draw = np.array(results_to_draw)
        plt.plot(range(test_data_num), labels_ref[:test_data_num, 0], color='r')
        plt.plot(range(results_to_draw.shape[0]), results_to_draw[:, 0, 0], color='b')
        plt.plot(range(test_data_num), labels_ref[:test_data_num, 1], color='g')
        plt.plot(range(results_to_draw.shape[0]), results_to_draw[:, 0, 1], color='m')
        plt.show()


