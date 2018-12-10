import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
total_data_num = 30
learning_rate = 2e-4
epoch_num = 1000
save_every_n_epoch = 50

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_xy": 64,
    "input1_dim_z": 36,
    "input2_dim": 8,
    "input3_dim": 8
}

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 20,
    "time_step": 4,
    "state_len": 64,
    "input_len": 576,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 512,  # should be the same as encoder out dim
    "dim2": 32,
    "dim3": 32  # dim1 + dim2 + dim3 should be input_len of the rnn, for line vector
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
    "stride_z2": 3,
    "channel2": 64,
    "kernel3": 3,
    "stride_xy3": 2,
    "stride_z3": 2,
    "channel3": 128,
    "out_dia": 2048
}

input_dimension_xy = input_paras["input1_dim_xy"]
input_dimension_z = input_paras["input1_dim_z"]


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


def relu_layer(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
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


def generate_sin_x_plus_y(number, side_dim_xy, side_dim_z, z_dim, out_dim, step, start_x, start_y, start_z):
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
        xyz = (math.sin(x+y) + math.cos(z)) / 2.0  # To (-1, 1)

        # data1
        cube = []
        for j in range(side_dim_xy):
            if j < side_dim_xy / 2:
                cube.append([[sx for m in range(side_dim_z)] for n in range(side_dim_xy)])
            else:
                cube.append([[sy for m in range(side_dim_z)] for n in range(side_dim_xy)])

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


if __name__ == '__main__':
    ''' Make some training values '''
    # total data number
    data_num = total_data_num

    print "generating data... "
    # create a dataset, validate
    data1, data2, label = generate_sin_x_plus_y(data_num, input_dimension_xy, input_dimension_z, input_paras["input2_dim"], rnn_paras["output_len"], 0.1, 0, 0.5, 2)
    data1 = data1.reshape(data_num, input_dimension_xy, input_dimension_xy, input_dimension_z, 1)

    print "Data generated!"

    ''' Graph building '''

    cube_data = tf.placeholder("float", name="cube_data", shape=[None, input_dimension_xy, input_dimension_xy, input_dimension_z, 1])
    line_data_1 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input2_dim"]])  # States
    line_data_2 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input3_dim"]])  # commands
    reference = tf.placeholder("float", name="reference", shape=[None, rnn_paras["output_len"]])

    # 3D CNN
    encode_vector = encoder(cube_data)
    # To flat vector
    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["out_dia"]])
    # Dropout 1
    encode_vector_flat = tf.layers.dropout(encode_vector_flat, rate=0.3, training=True)
    # Add a fully connected layer for map
    with tf.variable_scope("relu_encoder"):
        map_data_line = relu_layer(encode_vector_flat, encoder_para["out_dia"], concat_paras["dim1"])
    # Add a fully connected layer for states
    with tf.variable_scope("relu_states"):
        states_data_line = relu_layer(line_data_1, input_paras["input2_dim"], concat_paras["dim2"])
    # Add a fully connected layer for commands
    with tf.variable_scope("relu_commands"):
        commands_data_line = relu_layer(line_data_2, input_paras["input3_dim"], concat_paras["dim3"])
    # Concat, Note: dimension parameter should be 1, considering batch size
    concat_vector = tf.concat([map_data_line, states_data_line, commands_data_line], 1)
    # Dropout 2
    concat_vector = tf.layers.dropout(concat_vector, rate=0.3, training=True)
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
        restorer.restore(sess, "/home/clarence/log/model/40_autoencoder.ckpt")

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
                data1_batch = get_bacth_step(sequence[start_position:end_position], rnn_paras["time_step"], data1)
                data2_batch = get_bacth_step(sequence[start_position:end_position], rnn_paras["time_step"], data2)
                data3_batch = data2_batch
                label_batch = get_bacth(sequence[start_position:end_position], label)

                # train
                sess.run(train_step, feed_dict={cube_data: data1_batch, line_data_1: data2_batch, line_data_2: data3_batch, reference: label_batch})  # training

            print('loss for this epoch=%s' % sess.run(loss, feed_dict={cube_data: data1_batch, line_data_1: data2_batch, line_data_2: data3_batch, reference: label_batch}))

            if epoch % save_every_n_epoch == 0:
                # save
                saver.save(sess, '/home/clarence/log/model_rnn/'+'simulation_cnn_rnn'+str(epoch)+'.ckpt')


