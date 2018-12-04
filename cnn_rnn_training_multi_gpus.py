import tensorflow as tf
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import csv
import time
import gc

''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
total_data_num = 0  # Anyvalue
input_side_dimension = 64
learning_rate = 1e-4
epoch_num = 2000
save_every_n_epoch = 100

states_num_one_line = 11
labels_num_one_line = 4

path = "/home/ubuntu/chg_workspace/data/csvs"
clouds_filename = ["/chg_route1_trial1/pcl_data_2018_12_03_11:34:18.csv", "/hzy_route1_trial1/pcl_data_2018_12_03_11:26:38.csv"]
states_filename = ["/chg_route1_trial1/uav_data_2018_12_03_11:34:18.csv", "/hzy_route1_trial1/uav_data_2018_12_03_11:26:38.csv"]
labels_filename = ["/chg_route1_trial1/label_data_2018_12_03_11:34:18.csv", "/hzy_route1_trial1/label_data_2018_12_03_11:26:38.csv"]

img_wid = input_side_dimension
img_height = input_side_dimension

''' Parameters for Computer'''
gpu_num = 2

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 30,
    "time_step": 5,
    "state_len": 128,
    "input_len": 2304,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 2048,  # should be the same as encoder out dim
    "dim2": 256  # dim1 + dim2 should be input_len of the rnn, for line vector
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


def myrnn(x, input_len, output_len, raw_batch_size, time_step, state_len):
    """
    RNN function
    x: [raw_batch_size, time_step, input_len]
    state dimension is also weights dimension in hidden layer
    output_len can be given as you want(same as label dimension)
    """
    with tf.variable_scope("rnn"):
        w = tf.get_variable("weight_x", [input_len, state_len], initializer=tf.truncated_normal_initializer(stddev=0.1)) #tf.random_normal_initializer)
        u = tf.get_variable("weight_s", [state_len, state_len], initializer=tf.truncated_normal_initializer(stddev=0.1)) #tf.random_normal_initializer)
        v = tf.get_variable("weight_y", [state_len, output_len], initializer=tf.truncated_normal_initializer(stddev=0.1)) #tf.random_normal_initializer)
        b = tf.get_variable("bias", [output_len], initializer=tf.constant_initializer(0.0))

        state = tf.get_variable("state", [raw_batch_size, state_len], trainable=False, initializer=tf.constant_initializer(0.0))

        for seq in range(time_step):
            x_temp = x[:, seq, :]  # might not be right
            state = tf.nn.tanh(tf.matmul(state, u) + tf.matmul(x_temp, w))  # hidden layer activate function

        return tf.nn.tanh(tf.matmul(state, v) + b)  # output layer activate function


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


def generate_shuffled_array(start, stop, shuffle=True):
    """
    Give a length and return a shuffled one dimension array using data from start to stop, stop not included
    Used as shuffled sequence
    """
    array = np.arange(start, stop)
    if shuffle:
        np.random.shuffle(array)
    return array


def generate_shuffled_arrays_multifiles(start, stop, shuffle=True):
    """
    Give a length and return a shuffled one dimension array using data from start to stop, stop not included
    Used as shuffled sequence
    :param start: list of start positions in files, commonly (step-1)
    :param stop: list of end positions in files, commonly (data number)
    :param shuffle: if shuffle
    :return: one array
    """
    array = []
    start_this = 0
    stop_this = 0
    start_i = 0
    stop_i =0

    for i in range(len(start)):
        start_i = start_this + start[i]
        stop_i = stop_this + stop[i]

        array_this = np.arange(start_i, stop_i)
        if i == 0:
            array = array_this
        else:
            array = np.concatenate([array, array_this], axis=0)

        start_this = start_this + stop[i]
        stop_this = stop_this + stop[i]

    if shuffle:
        np.random.shuffle(array)
    return array


def get_bacth_step(seq, time_step, data):
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


def get_bacth(seq, data):
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
    print "Reading data..."

    data_mat = []
    states_input = []
    labels_ref = []

    data_num_in_files = []
    total_data_num = 0

    for file_seq in range(len(clouds_filename)):
        # Read clouds
        cloud_filename_this = str(path + clouds_filename[file_seq])
        clouds = open(cloud_filename_this, "r")
        img_num = len(clouds.readlines())
        clouds.close()
        data_mat_this = np.ones([img_num, img_wid, img_wid, img_height, 1])
        read_pcl(data_mat_this, cloud_filename_this)

        # Read states
        state_filename_this = str(path + states_filename[file_seq])
        states = open(state_filename_this, "r")
        states_num = len(states.readlines())
        states.close()
        if states_num != img_num:
            raise Networkerror("states file mismatch!")
        states_mat_this = np.zeros([states_num, states_num_one_line])
        read_others(states_mat_this, state_filename_this, states_num_one_line)

        # Read labels
        label_filename_this = str(path + labels_filename[file_seq])
        labels = open(label_filename_this, "r")
        labels_num = len(labels.readlines())
        labels.close()
        if labels_num != states_num:
            raise Networkerror("labels file mismatch!")
        labels_mat_this = np.zeros([labels_num, labels_num_one_line])
        read_others(labels_mat_this, label_filename_this, labels_num_one_line)

        ''' Choose useful states and labels '''
        compose_num = [200, 28, 28]
        # check total number
        num_total = 0
        for num_x in compose_num:
            num_total = num_total + num_x
        if num_total != concat_paras["dim2"]:
            raise Networkerror("compose_num does not match concat_paras!")
        # concat for input2
        states_input_delt_yaw = np.concatenate([np.reshape(states_mat_this[:, 10], [states_num, 1]) for i in range(compose_num[0])], axis=1)  # delt_yaw
        states_input_linear_vel = np.concatenate([np.reshape(states_mat_this[:, 2], [states_num, 1]) for i in range(compose_num[1])], axis=1)  # linear vel
        states_input_angular_vel = np.concatenate([np.reshape(states_mat_this[:, 3], [states_num, 1]) for i in range(compose_num[2])], axis=1)  # angular vel

        states_input_this = np.concatenate([states_input_delt_yaw, states_input_linear_vel, states_input_angular_vel], axis=1)

        labels_ref_this = labels_mat_this[:, 0:2]  # vel_cmd, angular_cmd

        total_data_num = total_data_num + img_num  # total pointclouds number
        data_num_in_files.append(img_num)

        # concat for all files
        if file_seq == 0:
            data_mat = data_mat_this
            states_input = states_input_this
            labels_ref = labels_ref_this
        else:
            data_mat = np.concatenate([data_mat, data_mat_this], axis=0)
            states_input = np.concatenate([states_input, states_input_this], axis=0)
            labels_ref = np.concatenate([labels_ref, labels_ref_this], axis=0)

    print "Data reading is completed!"

    ''' Calculate start and stop position for training list '''
    start_position_list = []
    stop_position_list = []

    for file_seq in range(len(clouds_filename)):
        start_position_list.append(rnn_paras["time_step"] - 1)
        stop_position_list.append(data_num_in_files[file_seq])

    ''' Calculate batch size '''
    batch_size_one_gpu = rnn_paras["raw_batch_size"]
    batch_size = batch_size_one_gpu * gpu_num
    batch_num = int((total_data_num - rnn_paras["time_step"] * len(clouds_filename)) / batch_size)  # issue

    ''' Graph building '''
    with tf.device("/cpu:0"):
        global_step = tf.train.get_or_create_global_step()
        tower_grads = []
        cube_data = tf.placeholder("float", name="cube_data", shape=[None, input_side_dimension, input_side_dimension, input_side_dimension, 1])
        line_data = tf.placeholder("float", name="line_data", shape=[None, concat_paras["dim2"]])
        reference = tf.placeholder("float", name="reference", shape=[None, rnn_paras["output_len"]])

        # Optimizer
        train_step = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_seq in range(gpu_num):
                with tf.device("/gpu:%d" % gpu_seq):
                    # Set data for each gpu
                    cube_data_this_gpu = cube_data[gpu_seq*batch_size_one_gpu*rnn_paras["time_step"]:(gpu_seq+1)*batch_size_one_gpu*rnn_paras["time_step"], :, :, :, :]
                    line_data_this_gpu = line_data[gpu_seq*batch_size_one_gpu*rnn_paras["time_step"]:(gpu_seq+1)*batch_size_one_gpu*rnn_paras["time_step"], :]
                    reference_this_gpu = reference[gpu_seq*batch_size_one_gpu:(gpu_seq+1)*batch_size_one_gpu, :]

                    # 3D CNN
                    encode_vector = encoder(cube_data_this_gpu)
                    # To flat vector
                    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["outdim"]])
                    # Concat, Note: dimension parameter should be 1, considering batch size
                    concat_vector = tf.concat([encode_vector_flat, line_data_this_gpu], 1)
                    # Dropout
                    # concat_vector = tf.layers.dropout(concat_vector, rate=0.3, training=True)
                    # Feed to rnn
                    rnn_input = tf.reshape(concat_vector, [batch_size_one_gpu, rnn_paras["time_step"], rnn_paras["input_len"]])
                    result_this_gpu = myrnn(rnn_input, rnn_paras["input_len"], rnn_paras["output_len"], batch_size_one_gpu, rnn_paras["time_step"], rnn_paras["state_len"])

                    tf.get_variable_scope().reuse_variables()
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
        print "Training data number = " + str(total_data_num)
        print "batch_size = " + str(batch_size)
        print "batch_num = " + str(batch_num)
        print "Total epoch num = " + str(epoch_num)
        print "Will save every " + str(save_every_n_epoch) + " epoches"

        # set restore and save parameters
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['encoder'])
        restorer = tf.train.Saver(variables_to_restore)
        variables_to_save = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
        saver = tf.train.Saver(variables_to_save)

        # Set memory filling parameters
        config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
        # config.gpu_options.allow_growth = True  # only 300M memory

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())  # initialze variables
            restorer.restore(sess, "/home/ubuntu/chg_workspace/3dcnn/model/simulation_autoencoder_450.ckpt")

            # start epoches
            for epoch in range(epoch_num):
                print "epoch: " + str(epoch)
                # get a random sequence for this epoch
                print "getting a random sequence for this epoch..."
                t0 = time.time()
                sequence = generate_shuffled_arrays_multifiles(start_position_list, stop_position_list, shuffle=True)

                # start batches
                for batch_seq in range(batch_num):
                    print "batch" + str(batch_seq)
                    # get data for this batch
                    start_position = batch_seq * batch_size
                    end_position = (batch_seq+1) * batch_size
                    data1_batch = get_bacth_step(sequence[start_position:end_position], rnn_paras["time_step"], data_mat)
                    data2_batch = get_bacth_step(sequence[start_position:end_position], rnn_paras["time_step"], states_input)
                    label_batch = get_bacth(sequence[start_position:end_position], labels_ref)

                    # train
                    sess.run(train_op, feed_dict={cube_data: data1_batch, line_data: data2_batch, reference: label_batch})  # training

                print('loss for this epoch=%s' % sess.run(loss, feed_dict={cube_data: data1_batch, line_data: data2_batch, reference: label_batch}))

                print "Time: " + str(time.time()-t0)

                if epoch % 1 == 0:
                    # get data for validation
                    start_position_test = random.randint(0, batch_num-1) * batch_size
                    end_position_test = start_position_test + batch_size
                    data1_batch_test = get_bacth_step(sequence[start_position_test:end_position_test], rnn_paras["time_step"],
                                                 data_mat)
                    data2_batch_test = get_bacth_step(sequence[start_position_test:end_position_test], rnn_paras["time_step"],
                                                 states_input)
                    label_batch_test = get_bacth(sequence[start_position_test:end_position_test], labels_ref)

                    # draw
                    results = sess.run(result_this_gpu, feed_dict={cube_data: data1_batch_test, line_data: data2_batch_test, reference: label_batch_test})
                    plt.plot(range(batch_size_one_gpu), results[:, 0], color='r')
                    plt.plot(range(batch_size_one_gpu), label_batch_test[:batch_size_one_gpu, 0], color='m')
                    plt.plot(range(batch_size_one_gpu), results[:, 1], color='g')
                    plt.plot(range(batch_size_one_gpu), label_batch_test[:batch_size_one_gpu, 1], color='b')
                    plt.show()

                if epoch != 0 and epoch % save_every_n_epoch == 0:
                    # save
                    saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/model_cnn_rnn_timestep5/'+'simulation_cnn_rnn'+str(epoch)+'.ckpt')


