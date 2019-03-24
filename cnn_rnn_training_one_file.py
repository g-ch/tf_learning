import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import csv
import time
import gc
import os
from multiprocessing import Pool
import multiprocessing


''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
learning_rate = 1e-4
epoch_num = 500
save_every_n_epoch = 50
training_times_simple_epoch = 2
if_train_encoder = True
if_continue_train = False

model_save_path = "/home/ubuntu/chg_workspace/3dcnn/model/cnn_rnn/01/model_onefile/"
image_save_path = "/home/ubuntu/chg_workspace/3dcnn/model/cnn_rnn/01/plot3/"

encoder_model = "/home/ubuntu/chg_workspace/3dcnn/model/auto_encoder/encoder_003/model/simulation_autoencoder_700.ckpt"
last_model = "/home/ubuntu/chg_workspace/3dcnn/model/cnn_rnn/01/model2/simulation_cnn_rnn50.ckpt"

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_xy": 64,  # point cloud
    "input1_dim_z": 24,  # point cloud
    "input2_dim": 8,  # states
    "input3_dim": 8  # commands
}

states_compose_num = [3, 3, 1, 1]  # total:  "input2_dim": 8
commands_compose_each = 2  # Should be "input3_dim": 8  / 4

input_dimension_xy = input_paras["input1_dim_xy"]
input_dimension_z = input_paras["input1_dim_z"]

img_wid = input_dimension_xy
img_height = input_dimension_z

''' Parameters for csv files '''

states_num_one_line = 17
labels_num_one_line = 4

file_path_states = "/home/ubuntu/chg_workspace/data/new_csvs/new_map/cnn-rnn/hzy_02/uav_data_2018_12_16_16:15:00.csv"

file_path_clouds = "/home/ubuntu/chg_workspace/data/new_csvs/new_map/cnn-rnn/hzy_02/pcl_data_2018_12_16_16:15:00.csv"

file_path_labels = "/home/ubuntu/chg_workspace/data/new_csvs/new_map/cnn-rnn/hzy_02/label_data_2018_12_16_16:15:00.csv"

''' Parameters for Computer'''
gpu_num = 2

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 20,
    "time_step": 10,
    "state_len": 16,
    "input_len": 592,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 512,  # should be the same as encoder out dim
    "dim2": 32,
    "dim3": 48  # dim1 + dim2 + dim3 should be input_len of the rnn, for line vector
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


def myrnn(x, input_len, output_len, raw_batch_size, time_step, state_len):
    """
    RNN function
    x: [raw_batch_size, time_step, input_len]
    state dimension is also weights dimension in hidden layer
    output_len can be given as you want(same as label dimension)
    """
    with tf.variable_scope("rnn"):
        w = tf.get_variable("weight_x", [input_len, state_len],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))  # tf.random_normal_initializer)
        u = tf.get_variable("weight_s", [state_len, state_len],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))  # tf.random_normal_initializer)
        v = tf.get_variable("weight_y", [state_len, output_len],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))  # tf.random_normal_initializer)
        b = tf.get_variable("bias", [output_len], initializer=tf.constant_initializer(0.0))

        state = tf.get_variable("state", [raw_batch_size, state_len], trainable=False,
                                initializer=tf.constant_initializer(0.0))

        for seq in range(time_step):
            x_temp = x[:, seq, :]  # might not be right
            state = tf.nn.tanh(tf.matmul(state, u) + tf.matmul(x_temp, w))  # hidden layer activate function

        return tf.nn.tanh(tf.matmul(state, v) + b)  # output layer activate function


def conv3d_relu(x, kernel_shape, bias_shape, strides):
    """ 3D convolution For 3D CNN encoder """
    ''' Parameters won't change in this training file '''
    weights = tf.get_variable("weights_con", kernel_shape, trainable=if_train_encoder)
    biases = tf.get_variable("bias_con", bias_shape, trainable=if_train_encoder)
    conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def max_pool(x, kernel_shape, strides):
    """ 3D convolution For 3D CNN encoder """
    return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu_layer(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights", [x_diamension, neurals_num],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias", [neurals_num], initializer=tf.constant_initializer(0.1))
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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


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


def draw_plots(x, y):
    """
    Draw multiple plots
    :param x: should be 2d array
    :param y: should be 2d array
    :return:
    """
    #for i in range(y.shape[0]):
    plt.plot(x, y)
    # plt.plot(x, y[1])

    plt.title("matplotlib")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)
    plt.show()


def generate_shuffled_array(start, stop, shuffle=True):
    """
    Give a length and return a shuffled one dimension array using data from start to stop, stop not included
    Used as shuffled sequence
    """
    array = np.arange(start, stop)
    if shuffle:
        np.random.shuffle(array)
    return array


def get_batch_step(seq, time_step, data):
    """
    get values of the seq in data(array), together with time_step back values
    :param seq: sequence to get, 0 or positive integers in one dimention array
    :param time_step: 2 at least
    :param data: data to get, must be numpy array!!!, at least 2 dimension
    :return: list [seq_size*time_step, data_size:] typical(if values in seq are all valid).
    """
    shape = list(data.shape)
    shape[0] = seq.shape[0] * time_step
    result = np.zeros(shape)
    step = time_step - 1

    gc.disable()

    for k in range(seq.shape[0]):
        for j in range(-step, 1, 1):
            result[k*time_step+step+j, :] = data[seq[k] + j, :]

    gc.enable()
    return result


def get_batch(seq, data):
    """
    get values of the seq in data(array), together with time_step back values
    :param seq: sequence to get, 0 or positive integers in one dimention array
    :param data: data to get, must be numpy array!!!, at least 2 dimension
    :return: list [seq_size*time_step, data_size:] typical(if values in seq are all valid).
    """
    shape = list(data.shape)
    shape[0] = seq.shape[0]
    result = np.zeros(shape)

    gc.disable()
    for k in range(seq.shape[0]):
        result[k, :] = data[seq[k], :]
    gc.enable()

    return result


def read_threading(filename_pcl, filename_state, filename_label, flags, house):
    """
    Read data thread function.
    :param filename_pcl:  pcl filename
    :param filename_state: state filename
    :param filename_label: label filename
    :param flags: flags to find a empty place
    :param house: house to store data, composed of [[[pcl], [state1], [state2], [label]],   [],   [],   []...]
    :return:
    """
    print "Start reading..."
    ''' Read pcl data first '''
    clouds = open(filename_pcl, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_pcl = np.zeros([img_num, img_wid, img_wid, img_height, 1])
    read_pcl(data_pcl, filename_pcl)
    print "pcl data get! img_num = " + str(img_num)

    # Just to make sure the data is read correctly
    # compare_draw_3d_to_2d(data_pcl[10, :, :, :, 0], data_pcl[10, :, :, :, 0], 0, 1, 2, 12, 1)

    ''' Read state data '''
    data_states = np.zeros([img_num, states_num_one_line])
    read_others(data_states, filename_state, states_num_one_line)

    ''' Read label data '''
    data_labels = np.zeros([img_num, labels_num_one_line])
    read_others(data_labels, filename_label, labels_num_one_line)

    ''' Get useful states and labels '''
    states_input_current_yaw_x = np.concatenate([np.reshape(data_states[:, 10], [img_num, 1])
                                                 for i in range(states_compose_num[0])], axis=1)  # current yaw x
    states_input_current_yaw_y = np.concatenate([np.reshape(data_states[:, 11], [img_num, 1])
                                                 for i in range(states_compose_num[1])], axis=1)  # current yaw y
    states_input_linear_vel = np.concatenate([np.reshape(data_states[:, 2], [img_num, 1])
                                              for i in range(states_compose_num[2])], axis=1)  # linear vel
    states_input_angular_vel = np.concatenate([np.reshape(data_states[:, 3], [img_num, 1])
                                               for i in range(states_compose_num[3])], axis=1)  # angular vel
    states_input = np.concatenate([states_input_current_yaw_x, states_input_current_yaw_y,
                                   states_input_linear_vel, states_input_angular_vel], axis=1)

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
    labels_ref[:, 1] = 0.8 * labels_ref[:, 1] #!!!!!

    ''' Store data to house '''
    looking_for_free_space_flag = True
    while looking_for_free_space_flag:
        time.sleep(0.05)
        for i_flag in range(len(flags)):
            if flags[i_flag] == 0:
                flags[i_flag] = 1
                print "found available space, copy data... "
                house[i_flag] = [data_pcl, states_input, commands_input, labels_ref]
                # label_to_draw = np.reshape(labels_ref[:, 0], [img_num])
                # draw_plots(np.arange(0, img_num), label_to_draw)
                flags[i_flag] = 2
                looking_for_free_space_flag = False
                break


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


def tf_training(data_read_flags, data_house, file_num):
    """
    Main training function
    :param data_read_flags: flag to find stored data
    :param data_house: where the data stores
    :param file_num: total file number of input csvs
    :return:
    """
    ''' Calculate batch size '''
    batch_size_one_gpu = rnn_paras["raw_batch_size"]
    batch_size = batch_size_one_gpu * gpu_num

    ''' Graph building '''
    print "Building graph!"
    with tf.device("/cpu:0"):
        global_step = tf.train.get_or_create_global_step()
        tower_grads = []
        cube_data = tf.placeholder("float", name="cube_data", shape=[None, input_dimension_xy, input_dimension_xy,
                                                                     input_dimension_z, 1])
        line_data_1 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input2_dim"]])  # States
        line_data_2 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input3_dim"]])  # commands
        reference = tf.placeholder("float", name="reference", shape=[None, rnn_paras["output_len"]])

        # Optimizer
        train_step = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_seq in range(gpu_num):
                with tf.device("/gpu:%d" % gpu_seq):
                    # Set data for each gpu
                    cube_data_this_gpu = cube_data[gpu_seq * batch_size_one_gpu * rnn_paras["time_step"]:
                                                   (gpu_seq + 1) * batch_size_one_gpu * rnn_paras["time_step"], :, :, :, :]
                    line_data_1_this_gpu = line_data_1[gpu_seq * batch_size_one_gpu * rnn_paras["time_step"]:
                                                       (gpu_seq + 1) * batch_size_one_gpu * rnn_paras["time_step"], :]
                    line_data_2_this_gpu = line_data_2[gpu_seq * batch_size_one_gpu * rnn_paras["time_step"]:
                                                       (gpu_seq + 1) * batch_size_one_gpu * rnn_paras["time_step"], :]
                    reference_this_gpu = reference[gpu_seq * batch_size_one_gpu:(gpu_seq + 1) * batch_size_one_gpu, :]

                    # 3D CNN
                    encode_vector = encoder(cube_data_this_gpu)
                    # To flat vector
                    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["out_dia"]])
                    # Dropout 1
                    encode_vector_flat = tf.layers.dropout(encode_vector_flat, rate=0.3, training=True)

                    # Add a fully connected layer for map
                    with tf.variable_scope("relu_encoder_1"):
                        map_data_line_0 = relu_layer(encode_vector_flat, encoder_para["out_dia"], concat_paras["dim1"])
                    with tf.variable_scope("relu_encoder_2"):
                        map_data_line = relu_layer(map_data_line_0, concat_paras["dim1"], concat_paras["dim1"])
                    # Add a fully connected layer for states
                    with tf.variable_scope("relu_states_1"):
                        states_data_line_0 = relu_layer(line_data_1_this_gpu, input_paras["input2_dim"],
                                                        concat_paras["dim2"])
                    with tf.variable_scope("relu_states_2"):
                        states_data_line = relu_layer(states_data_line_0, concat_paras["dim2"], concat_paras["dim2"])
                    # Add a fully connected layer for commands
                    with tf.variable_scope("relu_commands_1"):
                        commands_data_line_0 = relu_layer(line_data_2_this_gpu, input_paras["input3_dim"],
                                                          concat_paras["dim3"])
                    with tf.variable_scope("relu_commands_2"):
                        commands_data_line = relu_layer(commands_data_line_0, concat_paras["dim3"],
                                                        concat_paras["dim3"])

                    # Concat, Note: dimension parameter should be 1, considering batch size
                    concat_vector = tf.concat([map_data_line, states_data_line, commands_data_line], 1)

                    # Add a fully connected layer for all input before rnn
                    with tf.variable_scope("relu_all_1"):
                        relu_data_all = relu_layer(concat_vector, rnn_paras["input_len"],
                                                   rnn_paras["input_len"])
                    # Dropout 2
                    relu_data_droped = tf.layers.dropout(relu_data_all, rate=0.3, training=True)
                    # Feed to rnn
                    rnn_input = tf.reshape(relu_data_droped, [rnn_paras["raw_batch_size"], rnn_paras["time_step"],
                                                           rnn_paras["input_len"]])
                    result_this_gpu = myrnn(rnn_input, rnn_paras["input_len"], rnn_paras["output_len"],
                                            rnn_paras["raw_batch_size"], rnn_paras["time_step"], rnn_paras["state_len"])

                    tf.get_variable_scope().reuse_variables()

                    # Note!!! special loss
                    # temp_to_merge1 = tf.reshape(result_this_gpu[:, 0], [tf.shape(result_this_gpu)[0], 1])
                    # temp_to_merge2 = tf.zeros([tf.shape(result_this_gpu)[0], 1])
                    # result_merged = tf.concat([temp_to_merge1, temp_to_merge2], axis=1)  # keep linear velocity
                    # loss = tf.reduce_mean(tf.square(reference_this_gpu - result_this_gpu) + 0.2 * tf.square(
                    #     tf.abs(result_merged) - result_merged))  # expect a positive linear velocity
                    loss = tf.reduce_mean(tf.square(reference_this_gpu - result_this_gpu))

                    grads = train_step.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        train_op = train_step.apply_gradients(grads)

        ''' Show trainable variables '''
        variable_name = [v.name for v in tf.trainable_variables()]
        print "variable_name", variable_name

        ''' Training '''
        print "Start training"
        print "batch_size = " + str(batch_size)
        print "Total epoch num = " + str(epoch_num)
        print "Will save every " + str(save_every_n_epoch) + " epoches"

        # set restore and save parameters
        if if_continue_train:
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
            restorer = tf.train.Saver(variables_to_restore)  # optional
        else:
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['encoder'])
            restorer = tf.train.Saver(variables_to_restore)

        variables_to_save = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
        saver = tf.train.Saver(variables_to_save)

        # Set memory filling parameters
        config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
        # config.gpu_options.allow_growth = True  # only 300M memory

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())  # initialze variables
            if if_continue_train:
                restorer.restore(sess, last_model)
                print "Restored from last trained model !!"
            else:
                restorer.restore(sess, encoder_model)
                print "Partially restored from encoder !!"

            data_mat_pcl = np.array(0)
            data_mat_state = np.array(0)
            data_mat_command = np.array(0)
            data_mat_label = np.array(0)
            data_num = 0

            # Check flags to find data
            looking_for_data_flag = True
            while looking_for_data_flag:
                time.sleep(0.05)
                for i_flag in range(len(data_read_flags)):
                    if data_read_flags[i_flag] == 2:
                        print "found available data.. "
                        data_mat_pcl = data_house[i_flag][0]
                        data_mat_state = data_house[i_flag][1]
                        data_mat_command = data_house[i_flag][2]
                        data_mat_label = data_house[i_flag][3]
                        data_num = data_house[i_flag][3].shape[0]
                        # data_read_flags[i_flag] = 0
                        looking_for_data_flag = False
                        break
            print "done loading data.."

            # start epochs
            for epoch in range(epoch_num):
                print "epoch: " + str(epoch)
                t0 = time.time()

                ''' waiting for data '''

                batch_num = int((data_num - rnn_paras["time_step"]) / batch_size)

                for training_time_this_file in range(training_times_simple_epoch):
                    # get a random sequence for this file
                    sequence = generate_shuffled_array(rnn_paras["time_step"], data_num, shuffle=True)

                    # start batches
                    for batch_seq in range(batch_num):
                        print "batch" + str(batch_seq)
                        # get data for this batch
                        start_position = batch_seq * batch_size
                        end_position = (batch_seq + 1) * batch_size
                        data_pcl_batch = get_batch_step(sequence[start_position:end_position],
                                                        rnn_paras["time_step"], data_mat_pcl)
                        data_state_batch = get_batch_step(sequence[start_position:end_position],
                                                          rnn_paras["time_step"], data_mat_state)
                        data_command_batch = get_batch_step(sequence[start_position:end_position],
                                                            rnn_paras["time_step"], data_mat_command)
                        label_batch = get_batch(sequence[start_position:end_position], data_mat_label)

                        # label_to_draw = np.reshape(label_batch[:, 0], [batch_size])
                        # draw_plots(np.arange(0, batch_size), label_to_draw)

                        # train
                        sess.run(train_op, feed_dict={cube_data: data_pcl_batch, line_data_1: data_state_batch,
                                                      line_data_2: data_command_batch, reference: label_batch})

                        del data_pcl_batch
                        del data_state_batch
                        del data_command_batch
                        del label_batch

                print "Epoch " + str(epoch) + " finished!  Time: " + str(time.time() - t0)

                # if epoch % 10 == 0:
                #     # get data for validation
                #     start_position_test = random.randint(0, batch_num - 1) * batch_size
                #     end_position_test = start_position_test + batch_size
                #     data_pcl_batch_test = get_batch_step(sequence[start_position_test:end_position_test],
                #                                          rnn_paras["time_step"], data_mat_pcl)
                #     data_state_batch_test = get_batch_step(sequence[start_position_test:end_position_test],
                #                                            rnn_paras["time_step"], data_mat_state)
                #     data_command_batch_test = get_batch_step(sequence[start_position_test:end_position_test],
                #                                              rnn_paras["time_step"], data_mat_command)
                #     label_batch_test = get_batch(sequence[start_position_test:end_position_test], data_mat_label)
                #
                #     # draw
                #     results = sess.run(result_this_gpu,
                #                        feed_dict={cube_data: data_pcl_batch_test, line_data_1: data_state_batch_test,
                #                                   line_data_2: data_command_batch_test, reference: label_batch_test})
                #
                #     plt.plot(range(batch_size_one_gpu), results[:, 0], color='r')
                #     plt.plot(range(batch_size_one_gpu), label_batch_test[:batch_size_one_gpu, 0], color='m')
                #     plt.plot(range(batch_size_one_gpu), results[:, 1], color='g')
                #     plt.plot(range(batch_size_one_gpu), label_batch_test[:batch_size_one_gpu, 1], color='b')
                #     plt.savefig(image_save_path + str(epoch) + ".png")
                #     plt.close()
                #
                #     del data_pcl_batch_test
                #     del data_state_batch_test
                #     del data_command_batch_test
                #     del label_batch_test

                if epoch % save_every_n_epoch == 0:
                    # save
                    saver.save(sess, model_save_path + "simulation_cnn_rnn" + str(epoch) + ".ckpt")


if __name__ == '__main__':

    '''Multiple thread'''
    pool = Pool(processes=2)

    data_read_flags = multiprocessing.Manager().list([0, 0, 0, 0])
    data_house = multiprocessing.Manager().list([0, 0, 0, 0])

    # Training thread
    files_num = len(file_path_clouds)
    pool.apply_async(tf_training, args=(data_read_flags, data_house, files_num))

    # pool.apply_async(test)
    filename_pcl_this = file_path_clouds
    filename_states_this = file_path_states
    filename_labels_this = file_path_labels

    pool.apply_async(read_threading, args=(filename_pcl_this, filename_states_this, filename_labels_this,
                                                   data_read_flags, data_house))

    pool.close()
    pool.join()

