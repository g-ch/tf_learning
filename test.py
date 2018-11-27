import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 100,
    "time_step": 5,
    "state_dia": 512,
    "input_dia": 2,
    "output_dia": 2
}


def myrnn(x, input_len, output_len, raw_batch_size, time_step, state_dia):
    """
        RNN function
        x: [raw_batch_size, time_step, input_len]
        state diamension is also weights diamension in hidden layer
        output_len can be given as you want(same as label diamension)
    """
    w = tf.get_variable("weight_x", [input_len, state_dia], initializer=tf.truncated_normal_initializer(stddev=0.1))
    u = tf.get_variable("weight_s", [state_dia, state_dia], initializer=tf.truncated_normal_initializer(stddev=0.1))
    v = tf.get_variable("weight_y", [state_dia, output_len], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable("bias", [output_len], initializer=tf.constant_initializer(0.0))

    state = tf.get_variable("state", [raw_batch_size, state_dia], trainable=False, initializer=tf.constant_initializer(0.0))

    for seq in range(time_step):
        x_temp = x[:, seq, :]  # might not be right
        state = tf.nn.tanh(tf.matmul(state, u) + tf.matmul(x_temp, w))  # hidden layer activate function

    return tf.nn.tanh(tf.matmul(state, v) + b)  # output layer activate function


def generate(seq):
    X = []
    X_1 = []
    y = []
    y_1 = []
    for i in range(len(seq) - rnn_paras["time_step"] + 1):
        X.append(np.sin(seq[i:i + rnn_paras["time_step"]]))
        X_1.append(np.sin(seq[i:i + rnn_paras["time_step"]]))
        y.append(np.cos(seq[i + rnn_paras["time_step"] - 1]))
        y_1.append(np.cos(seq[i + rnn_paras["time_step"] - 1]))

    X = np.array(X, dtype=np.float32)
    X_1 = np.array(X_1, dtype=np.float32)
    X = np.reshape(X, [-1, 1])
    X_1 = np.reshape(X_1, [-1, 1])

    y = np.array(y, dtype=np.float32)
    y = np.reshape(y, [-1, 1])
    y_1 = np.array(y_1, dtype=np.float32)
    y_1 = np.reshape(y_1, [-1, 1])

    return np.hstack([X, X_1]), np.hstack([y, y_1])


if __name__ == '__main__':
    ''' Make some training values '''
    '''Random data generation'''
    # choice_value = [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0]  # from 0/7 to 7/7
    #
    # # let one batch generated to test first
    # data_num = rnn_paras["raw_batch_size"] * rnn_paras["time_step"]
    #
    # print "generating data... "
    # dia = rnn_paras["input_dia"]
    #
    # data_mat = [[choice_value[random.randint(0, 7)] for k in range(dia)] for h in range(data_num)]
    #
    # label_mat = []
    # for i in range(rnn_paras["raw_batch_size"]):
    #     mean = np.mean(data_mat[i*rnn_paras["time_step"]])  # start line mean
    #     for j in range(1, rnn_paras["time_step"]):
    #         mean = mean + np.mean(data_mat[i*rnn_paras["time_step"] + j])
    #     # The mean and the last data of the closest time matters here
    #     temp_value = mean / rnn_paras["time_step"] + data_mat[(i+1)*rnn_paras["time_step"]-1][rnn_paras["input_dia"]-1]
    #     label_mat.append([temp_value, temp_value])

    seq_train = np.linspace(start=0, stop=10, num=104, dtype=np.float32)
    data_raw = np.cos(seq_train)
    data_mat, label_mat = generate(seq_train)
    plt.plot(range(len(seq_train) - 4), data_raw[4:], 'r*')
    # data_mat = np.reshape(data_mat, [-1, 1])
    # label_mat = np.reshape(label_mat, [-1, 1])

    print "Data genarated!"

    ''' Graph building '''
    x_ = tf.placeholder("float", shape=[None, rnn_paras["input_dia"]])
    y_ = tf.placeholder("float", shape=[None, rnn_paras["output_dia"]])

    x_shaped = tf.reshape(x_, [rnn_paras["raw_batch_size"], rnn_paras["time_step"], rnn_paras["input_dia"]])

    with tf.variable_scope("rnn"):
        result = myrnn(x_shaped, rnn_paras["input_dia"], rnn_paras["output_dia"], rnn_paras["raw_batch_size"], rnn_paras["time_step"], rnn_paras["state_dia"])

    ''' Optimizer '''
    learning_rate = 1e-4
    loss = tf.reduce_mean(tf.square(y_ - result))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    ''' Show variables '''
    variable_name = [v.name for v in tf.trainable_variables()]
    print "variable_name", variable_name

    ''' Training '''
    print "Start training"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1000):

            sess.run(train_step, feed_dict={x_: data_mat, y_: label_mat})  # training

            print "epoch: " + str(epoch)
            print('loss=%s' % sess.run(loss, feed_dict={x_: data_mat, y_: label_mat}))

        temp = sess.run(result, feed_dict={x_: data_mat, y_: label_mat})
        plt.plot(range(100), temp[:, 0])
        plt.plot(range(100), temp[:, 1])
    plt.show()
