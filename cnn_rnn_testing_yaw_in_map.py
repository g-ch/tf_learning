import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import csv


''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
test_data_num = 200

states_num_one_line = 17
labels_num_one_line = 4

commands_compose_each = 1  # Should be "input3_dim": 8  / 4

model_path = "/home/ubuntu/chg_workspace/3dcnn_yaw_in_map/model/cnn_rnn/02/third_train/simulation_cnn_rnn150.ckpt"

clouds_filename = \
    "/home/ubuntu/chg_workspace/data/yaw_in_map/test/pcl_data_2018_12_28_11:26:40_chg_abnormal.csv"
states_filename = \
    "/home/ubuntu/chg_workspace/data/yaw_in_map/test/uav_data_2018_12_28_11:26:40_chg_abnormal.csv"
labels_filename = \
    "/home/ubuntu/chg_workspace/data/yaw_in_map/test/label_data_2018_12_28_11:26:40_chg_abnormal.csv"

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_xy": 64,  # point cloud
    "input1_dim_z": 24,  # point cloud
    "input3_dim": 4  # commands
}
input_dimension_xy = input_paras["input1_dim_xy"]
input_dimension_z = input_paras["input1_dim_z"]
img_wid = input_dimension_xy
img_height = input_dimension_z

''' Parameters for RNN'''
rnn_paras = {
    "state_len": 16,
    "input_len": 544,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 512,  # should be the same as encoder out dim
    "dim3": 32  # dim1 + dim3 should be input_len of the rnn, for line vector
}

''' Parameters for CNN encoder'''
encoder_para = {
    "kernel1": 5,
    "stride_xy1": 2,
    "stride_z1": 3,
    "channel1": 32,
    "pool1": 2,
    "kernel2": 3,
    "stride_xy2": 2,
    "stride_z2": 2,
    "channel2": 64,
    "kernel3": 3,
    "stride_xy3": 2,
    "stride_z3": 2,
    "channel3": 128,
    "out_dia": 2048
}


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
    ''' Parameters won't change in this training file '''
    weights = tf.get_variable("weights_con", kernel_shape)
    biases = tf.get_variable("bias_con", bias_shape)
    conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def max_pool(x, kernel_shape, strides):
    """ 3D convolution For 3D CNN encoder """
    return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu_layer(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights", [x_diamension, neurals_num])
    biases = tf.get_variable("bias", [neurals_num])
    return tf.nn.relu(tf.matmul(x, weights) + biases)


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
                                data[i_row, i, j, k, 0] = row[i * img_wid * img_height + j * img_height + k]
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


def draw_plots(x, y):
    """
    Draw multiple plots
    :param x: should be 2d array
    :param y: should be 2d array
    :return:
    """
    plt.plot(x, y)

    plt.title("matplotlib")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)
    plt.show()


# draw by axis z direction
def compare_draw_3d_to_2d(data1, data2, min_val, max_val, rows, cols, step):
    """
    To compare two 3 dimension array by image slices
    :param data1: data to compare, 3 dimension array
    :param data2: should have the same size as data1
    :param min_val: minimum value in data1 and data2
    :param max_val: maximum value in data1 and data2
    :param rows: row number of the figure
    :param cols: col number of the figure
    :param step: step in z axis to show
    :return:
    """
    colors = ['purple', 'yellow']
    bounds = [min_val, max_val]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    f, a = plt.subplots(rows, cols, figsize=(cols, rows))

    # for scale
    data1_copy = np.array(tuple(data1))
    data2_copy = np.array(tuple(data2))
    data1_copy[0, 0, :] = min_val
    data2_copy[0, 0, :] = min_val
    data1_copy[0, 1, :] = max_val
    data2_copy[0, 1, :] = max_val

    for i in range(cols):
        for j in range(rows / 2):
            a[2 * j][i].imshow(data1_copy[:, :, (j * cols + i) * step])
            a[2 * j + 1][i].imshow(data2_copy[:, :, (j * cols + i) * step])

    plt.show(cmap=cmap, norm=norm)


if __name__ == '__main__':
    ''' Make some training values '''
    print "Reading data..."
    # Read clouds
    clouds = open(clouds_filename, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_mat = np.ones([img_num, img_wid, img_wid, img_height, 1])
    read_pcl(data_mat, clouds_filename)

    # Just to make sure the data is read correctly
    print "data_mat shape = ", data_mat.shape
    compare_draw_3d_to_2d(data_mat[100, :, :, :, 0], data_mat[200, :, :, :, 0], 0, 1, 4, 12, 1)

    # Read states
    states = open(states_filename, "r")
    states_num = len(states.readlines())
    states.close()
    if states_num != img_num:
        raise Networkerror("states file mismatch!")
    data_states = np.zeros([states_num, states_num_one_line])
    read_others(data_states, states_filename, states_num_one_line)

    # Read labels
    labels = open(labels_filename, "r")
    labels_num = len(labels.readlines())
    labels.close()
    if labels_num != states_num:
        raise Networkerror("labels file mismatch!")
    data_labels = np.zeros([labels_num, labels_num_one_line])
    read_others(data_labels, labels_filename, labels_num_one_line)

    ''' Choose useful states and labels '''
    # concat for input2
    commands_input_forward = np.concatenate([np.reshape(data_states[:, 13], [img_num, 1])
                                             for i in range(commands_compose_each)], axis=1)  # command: forward
    commands_input_backward = np.concatenate([np.reshape(data_states[:, 14], [img_num, 1])
                                              for i in range(commands_compose_each)], axis=1)  # command: backward
    commands_input_left = np.concatenate([np.reshape(data_states[:, 15], [img_num, 1])
                                          for i in range(commands_compose_each)], axis=1)  # command: left
    commands_input_right = np.concatenate([np.reshape(data_states[:, 16], [img_num, 1])
                                           for i in range(commands_compose_each)], axis=1)  # command: right
    commands_input = np.concatenate([commands_input_forward, commands_input_backward,
                                     commands_input_left, commands_input_right], axis=1)

    labels_ref = data_labels[:, 0:2]  # vel_cmd ref, angular_cmd ref
    labels_ref[:, 1] = 0.8 * labels_ref[:, 1]  # !!!!!

    print "Data reading is completed!"

    ''' Graph building '''
    cube_data = tf.placeholder("float", name="cube_data", shape=[None, input_dimension_xy, input_dimension_xy,
                                                                 input_dimension_z, 1])
    line_data_2 = tf.placeholder("float", name="line_data_2", shape=[None, input_paras["input3_dim"]])  # commands
    state_data = tf.placeholder("float", name="state_data", shape=[1, rnn_paras["state_len"]])

    # 3D CNN
    encode_vector = encoder(cube_data)
    # To flat vector
    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["out_dia"]])

    # Add a fully connected layer for map
    with tf.variable_scope("relu_encoder_1"):
        map_data_line_0 = relu_layer(encode_vector_flat, encoder_para["out_dia"], concat_paras["dim1"])
    with tf.variable_scope("relu_encoder_2"):
        map_data_line = relu_layer(map_data_line_0, concat_paras["dim1"], concat_paras["dim1"])
    # Add a fully connected layer for commands
    with tf.variable_scope("relu_commands_1"):
        commands_data_line_0 = relu_layer(line_data_2, input_paras["input3_dim"],
                                          concat_paras["dim3"])
    with tf.variable_scope("relu_commands_2"):
        commands_data_line = relu_layer(commands_data_line_0, concat_paras["dim3"],
                                        concat_paras["dim3"])

    # Concat, Note: dimension parameter should be 1, considering batch size
    concat_vector = tf.concat([map_data_line, commands_data_line], 1)

    # Add a fully connected layer for all input before rnn
    with tf.variable_scope("relu_all_1"):
        relu_data_all = relu_layer(concat_vector, rnn_paras["input_len"],
                                   rnn_paras["input_len"])
    # Feed to rnn
    result, state_returned = myrnn_test(relu_data_all, state_data, rnn_paras["input_len"], rnn_paras["output_len"], rnn_paras["state_len"])

    ''' Predicting '''
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
    restorer = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, model_path)
        state_data_give = np.zeros([1, rnn_paras["state_len"]])

        results_to_draw = []
        for i in range(test_data_num):
            data1_to_feed = data_mat[i, :].reshape([1, input_dimension_xy, input_dimension_xy, input_dimension_z, 1])
            data3_to_feed = commands_input[i, :].reshape([1, input_paras["input3_dim"]])

            results = sess.run(result, feed_dict={cube_data: data1_to_feed, line_data_2: data3_to_feed, state_data: state_data_give})
            state_data_give = sess.run(state_returned, feed_dict={cube_data: data1_to_feed, line_data_2: data3_to_feed, state_data: state_data_give})

            if i % 10 == 0:
                concat_vector_to_show = sess.run(concat_vector, feed_dict={cube_data: data1_to_feed,
                                         line_data_2: data3_to_feed, state_data: state_data_give})
                draw_plots(np.arange(0, rnn_paras["input_len"]), np.reshape(concat_vector_to_show, [rnn_paras["input_len"]]))

            results_to_draw.append(results)
            print "result: ", results, "label: ", labels_ref[i]

        results_to_draw = np.array(results_to_draw)
        plt.plot(range(test_data_num), labels_ref[:test_data_num, 0], color='r')
        plt.plot(range(results_to_draw.shape[0]), results_to_draw[:, 0, 0], color='b')
        plt.plot(range(test_data_num), labels_ref[:test_data_num, 1], color='g')
        plt.plot(range(results_to_draw.shape[0]), results_to_draw[:, 0, 1], color='m')
        plt.show()

