import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
total_data_num = 0
input_side_dimension = 64
learning_rate = 1e-4
epoch_num = 1000
save_every_n_epoch = 20

states_num_one_line = 11
labels_num_one_line = 4

clouds_filename = "/home/ubuntu/chg_workspace/data/csvs/1_1/pcl_data_2018_12_01_11:03:07.csv"
states_filename = "/home/ubuntu/chg_workspace/data/csvs/1_1/uav_data_2018_12_01_11:03:07.csv"
labels_filename = "/home/ubuntu/chg_workspace/data/csvs/1_1/label_data_2018_12_01_11:03:07.csv"

img_wid = input_side_dimension
img_height = input_side_dimension

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 20,
    "time_step": 4,
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


def generate_sin_x_plus_y(number, side_dim, z_dim, out_dim, step, start_x, start_y, start_z):
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
    z = start_z

    data1 = []
    data2 = []
    label = []

    for i in range(number):
        sx = math.sin(x)
        sy = math.sin(y)
        sz = math.sin(z)
        xyz = (math.sin(x+y) + math.cos(z)) / 2.0

        # data1
        cube = []
        for j in range(side_dim):
            if j < side_dim / 2:
                cube.append([[sx for m in range(side_dim)] for n in range(side_dim)])
            else:
                cube.append([[sy for m in range(side_dim)] for n in range(side_dim)])

        data1.append(cube)
        # data2
        data2.append([sz for k in range(z_dim)])
        # label
        label.append([xyz*g for g in range(1, out_dim+1)])
        # update seed
        x = x + step
        y = y + step
        z = z + step

    return np.array(data1), np.array(data2), np.array(label)


def generate_shuffled_array(start, stop, shuffle=True):
    """
    Give a length and return a shuffled one dimension array using data from start to stop, stop not included
    Used as shuffled sequence
    """
    array = np.arange(start, stop)
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
    result = []
    step = time_step - 1
    for i in range(seq.shape[0]):
        for j in range(-step, 1, 1):
            result.append(data[seq[i] + j, :].tolist())

    return np.array(result)


def get_bacth(seq, data):
    """
    get values of the seq in data(array), together with time_step back values
    :param seq: sequence to get, 0 or positive integers in one dimention array
    :param data: data to get, must be numpy array!!!, at least 2 dimension
    :return: list [seq_size*time_step, data_size:] typical(if values in seq are all valid).
    """
    result = []
    for i in range(seq.shape[0]):
        result.append(data[seq[i], :].tolist())

    return np.array(result)


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
    compose_num = [256]
    # check total number
    num_total = 0
    for num_x in compose_num:
        num_total = num_total + num_x
    if num_total != concat_paras["dim2"]:
        raise Networkerror("compose_num does not match concat_paras!")
    # concat for input2
    states_input = np.concatenate([np.reshape(states_mat[:, 10], [states_num, 1]) for i in range(compose_num[0])], axis=1)  # vel_odom

    labels_ref = labels_mat[:, 0:2]  # vel_cmd, angular_cmd

    total_data_num = img_num

    print "Data reading is completed!"

    ''' Graph building '''

    cube_data = tf.placeholder("float", name="cube_data", shape=[None, input_side_dimension, input_side_dimension, input_side_dimension, 1])
    line_data = tf.placeholder("float", name="line_data", shape=[None, concat_paras["dim2"]])
    reference = tf.placeholder("float", name="reference", shape=[None, rnn_paras["output_len"]])

    # 3D CNN
    encode_vector = encoder(cube_data)
    # To flat vector
    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["outdim"]])
    # Concat, Note: dimension parameter should be 1, considering batch size
    concat_vector = tf.concat([encode_vector_flat, line_data], 1)
    # Dropout
    # concat_vector = tf.layers.dropout(concat_vector, rate=0.3, training=True)
    # Feed to rnn
    rnn_input = tf.reshape(concat_vector, [rnn_paras["raw_batch_size"], rnn_paras["time_step"], rnn_paras["input_len"]])
    result = myrnn(rnn_input, rnn_paras["input_len"], rnn_paras["output_len"], rnn_paras["raw_batch_size"], rnn_paras["time_step"], rnn_paras["state_len"])

    ''' Optimizer '''
    loss = tf.reduce_mean(tf.square(reference - result))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    ''' Show trainable variables '''
    variable_name = [v.name for v in tf.trainable_variables()]
    print "variable_name", variable_name

    ''' Training '''
    print "Start training"

    batch_size = rnn_paras["raw_batch_size"]
    batch_num = int((total_data_num - rnn_paras["time_step"]) / batch_size)  # issue

    # set restore and save parameters
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['encoder'])
    restorer = tf.train.Saver(variables_to_restore)
    variables_to_save = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
    saver = tf.train.Saver(variables_to_save)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, "/home/ubuntu/chg_workspace/3dcnn/model/simulation_autoencoder_450.ckpt")

        # start epoches
        for epoch in range(epoch_num):
            print "epoch: " + str(epoch)
            # get a random sequence for this epoch
            sequence = generate_shuffled_array(rnn_paras["time_step"] - 1, total_data_num, shuffle=True)
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
                sess.run(train_step, feed_dict={cube_data: data1_batch, line_data: data2_batch, reference: label_batch})  # training

            print('loss for this epoch=%s' % sess.run(loss, feed_dict={cube_data: data1_batch, line_data: data2_batch, reference: label_batch}))

            if epoch % 2 == 0:
                # draw
                results = sess.run(result, feed_dict={cube_data: data1_batch, line_data: data2_batch, reference: label_batch})
                plt.plot(range(results.shape[0]), results[:, 0])
                plt.plot(range(label_batch.shape[0]), label_batch[:, 0])
                plt.show()

            if epoch % save_every_n_epoch == 0:
                # save
                saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/model_cnn_rnn_timestep5/'+'simulation_cnn_rnn'+str(epoch)+'.ckpt')


