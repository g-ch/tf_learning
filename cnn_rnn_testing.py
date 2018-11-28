import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
test_data_num = 24
input_side_dimension = 64
learning_rate = 1e-4
epoch_num = 1000

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 20,
    "time_step": 4,
    "state_len": 128,
    "input_len": 2176,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 2048,  # should be the same as encoder out dim
    "dim2": 128  # dim1 + dim2 should be input_len of the rnn, for line vector
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
        xyz = math.sin(x+y) + math.cos(z)

        # data1
        cube = []
        for j in range(side_dim):
            if j % 2 == 0:
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


if __name__ == '__main__':
    ''' Make some training values '''
    # total data number
    data_num = test_data_num

    print "generating data... "
    cube_dim = input_side_dimension

    # create a dataset, validate
    data1, data2, label = generate_sin_x_plus_y(data_num, cube_dim, concat_paras["dim2"], rnn_paras["output_len"], 0.1, 0, 0.5, 2)
    data1 = data1.reshape(data_num, cube_dim, cube_dim, cube_dim, 1)

    # batch get example
    # batch_size = rnn_paras["raw_batch_size"]
    # data1_batch = get_bacth(sequence[0:batch_size], rnn_paras["time_step"], data1)
    # data2_batch = get_bacth(sequence[0:batch_size], rnn_paras["time_step"], data2)
    # label_batch = get_bacth(sequence[0:batch_size], 1, label)

    print "Data generated!"

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, "/home/ubuntu/chg_workspace/3dcnn/model/model_cnn_rnn_timestep5/70_cnn_rnn_2.ckpt")
        state_data_give = np.zeros([1, rnn_paras["state_len"]])

        results_to_draw = []
        for i in range(test_data_num):
            data1_to_feed = data1[i, :].reshape([1, cube_dim, cube_dim, cube_dim, 1])
            data2_to_feed = data2[i, :].reshape([1, concat_paras["dim2"]])

            results = sess.run(result, feed_dict={cube_data: data1_to_feed, line_data: data2_to_feed, state_data: state_data_give})
            state_data_give = sess.run(state_returned, feed_dict={cube_data: data1_to_feed, line_data: data2_to_feed, state_data: state_data_give})

            results_to_draw.append(results)
            print "result: ", results, "label: ", label[i]

        results_to_draw = np.array(results_to_draw)
        plt.plot(range(label.shape[0]), label[:, 0])
        plt.plot(range(results_to_draw.shape[0]), results_to_draw[:, 0, 0])
        plt.show()


