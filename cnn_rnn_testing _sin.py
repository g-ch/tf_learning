import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
test_data_num = 30

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_xy": 64,
    "input1_dim_z": 36,
    "input2_dim": 8,
    "input3_dim": 8
}

''' Parameters for RNN'''
rnn_paras = {
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
        xyz = (math.sin(x+y) + math.cos(z) + 2.0) / 2.0  # To (0, 1)

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


if __name__ == '__main__':
    ''' Make some training values '''
    # total data number
    data_num = test_data_num

    print "generating data... "
    # create a dataset, validate
    data1, data2, label = generate_sin_x_plus_y(data_num, input_dimension_xy, input_dimension_z,
                                                input_paras["input2_dim"], rnn_paras["output_len"], 0.4, 0.3, 0.8, 1)
    data1 = data1.reshape(data_num, input_dimension_xy, input_dimension_xy, input_dimension_z, 1)

    print "Data generated!"


    ''' Graph building '''
    cube_data = tf.placeholder("float", name="cube_data",
                               shape=[None, input_dimension_xy, input_dimension_xy, input_dimension_z, 1])
    line_data_1 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input2_dim"]])  # States
    line_data_2 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input3_dim"]])  # commands
    state_data = tf.placeholder("float", name="line_data", shape=[1, rnn_paras["state_len"]])

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

    result, state_returned = myrnn_test(concat_vector, state_data, rnn_paras["input_len"], rnn_paras["output_len"],
                                        rnn_paras["state_len"])

    ''' Predicting '''
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
    restorer = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, "/home/clarence/log/model_rnn/simulation_cnn_rnn500.ckpt")
        state_data_give = np.zeros([1, rnn_paras["state_len"]])

        results_to_draw = []
        for i in range(test_data_num):
            data1_to_feed = data1[i, :].reshape([1, input_dimension_xy, input_dimension_xy, input_dimension_z, 1])
            data2_to_feed = data2[i, :].reshape([1, input_paras["input2_dim"]])
            data3_to_feed = data2_to_feed

            results = sess.run(result, feed_dict={cube_data: data1_to_feed, line_data_1: data2_to_feed, line_data_2: data3_to_feed, state_data: state_data_give})
            state_data_give = sess.run(state_returned, feed_dict={cube_data: data1_to_feed, line_data_1: data2_to_feed, line_data_2: data3_to_feed, state_data: state_data_give})

            results_to_draw.append(results)
            print "result: ", results, "label: ", label[i]

        results_to_draw = np.array(results_to_draw)
        plt.plot(range(label.shape[0]), label[:, 0])
        plt.plot(range(results_to_draw.shape[0]), results_to_draw[:, 0, 0])
        plt.show()


