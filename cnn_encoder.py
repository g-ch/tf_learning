import tensorflow as tf
import numpy as np
import math
import sys
import csv

input_side_diamension = 64
batch_size = 20
learning_rate = 1e-4
total_epoches = 500
save_every_n_epoch = 50
filename = "/home/ubuntu/chg_workspace/data/csvs/pcl_data_2018_11_30_21:07:37.csv"

img_wid = input_side_diamension
img_height = input_side_diamension

device = {
    "gpu1": "0",
    "gpu2": "0"
}

encoder_para = {
    "kernel1": 5,
    "stride1": 2,  # do not change
    "channel1": 32,
    "kernel2": 3,
    "stride2": 1,  # do not change
    "channel2": 32,
    "pool1": 4,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "pool2": 2,  # 2 or 4 recommended, carefully choose according to input_side_diamension to keep values interger
    "outdia": 2048
}

pooled_side_len1 = input_side_diamension
pooled_side_len2 = int(input_side_diamension / (encoder_para["pool1"] * encoder_para["stride1"]))

pooled_size = int(input_side_diamension * input_side_diamension * input_side_diamension * encoder_para["channel2"] / math.pow(encoder_para["pool1"]*encoder_para["pool2"]*encoder_para["stride1"], 3))


def conv3d_relu(x, kernel_shape, bias_shape, strides):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.0))
        with tf.device("/gpu:"+device["gpu2"]):
            conv = tf.nn.conv3d(x, weights, strides=strides, padding="SAME")
            return tf.nn.relu(conv + biases)


def deconv3d(x, kernel_shape, output_shape, strides):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        with tf.device("/gpu:" + device["gpu1"]):
            return tf.nn.conv3d_transpose(x, filter=weights, output_shape=output_shape, strides=strides)


def max_pool(x, kernel_shape, strides):
    with tf.device("/gpu:" + device["gpu1"]):
        return tf.nn.max_pool3d(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu(x, x_diamension, neurals_num):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_relu", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias_relu", [neurals_num], initializer=tf.constant_initializer(0.0))
        with tf.device("/gpu:" + device["gpu1"]):
            return tf.nn.relu(tf.matmul(x, weights) + biases)


def softmax(x, x_diamension, neurals_num):
    with tf.device("/cpu:0"):
        weights = tf.get_variable("weights_soft", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias_soft", [neurals_num], initializer=tf.constant_initializer(0.0))
        with tf.device("/gpu:" + device["gpu1"]):
            return tf.nn.softmax(tf.matmul(x, weights) + biases)


def encoder(x):
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
            print "max_pool2 ", max_pool2
            return max_pool2


def decoder(x, batch_size):
    k2 = encoder_para["kernel2"]
    s2 = encoder_para["stride2"]
    d2 = encoder_para["channel2"]
    p2 = encoder_para["pool2"]

    k1 = encoder_para["kernel1"]
    s1 = encoder_para["stride1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"] * s1

    pl1 = pooled_side_len1
    pl2 = pooled_side_len2

    with tf.variable_scope("decoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv3d_relu(x, [k2, k2, k2, d2, d2], [d2], [1, 1, 1, 1, 1])
            print "conv1 ", conv1

        with tf.variable_scope("deconv1"):
            deconv1 = deconv3d(conv1, [p2, p2, p2, d2, d2], output_shape=[batch_size, pl2, pl2, pl2, d2], strides=[1, p2, p2, p2, 1])
            print "deconv1", deconv1.get_shape()
        with tf.variable_scope("conv2"):
            conv2 = conv3d_relu(deconv1, [k2, k2, k2, d2, d1], [d1], [1, 1, 1, 1, 1])

        with tf.variable_scope("deconv2"):
            deconv2 = deconv3d(conv2, [p1, p1, p1, d1, d1], output_shape=[batch_size, pl1, pl1, pl1, d1], strides=[1, p1, p1, p1, 1])
            print "deconv2", deconv2.get_shape()
        with tf.variable_scope("conv3"):
            conv3 = conv3d_relu(deconv2, [k1, k1, k1, d1, 1], [1], [1, 1, 1, 1, 1])
            print "conv3", conv3.get_shape()
            return conv3


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


def read_pcl(data,filename):
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


if __name__ == '__main__':
    '''Data reading'''
    print "Reading data..."
    clouds = open(filename, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_mat = np.ones([img_num, img_wid, img_wid, img_height, 1])
    read_pcl(data_mat, filename)

    print "Data reading is completed!"

    '''Training'''
    dia = input_side_diamension
    x_ = tf.placeholder("float", shape=[None, dia, dia, dia, 1])

    encode_vector = encoder(x_)

    print "encode_vector: ", encode_vector.get_shape()
    decode_result = decoder(encode_vector, batch_size)

    loss = tf.reduce_mean(tf.square(x_ - decode_result))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print "Start training"
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        total_data_num = data_mat.shape[0]
        print "get " + str(total_data_num) + " data"

        for epoch in range(total_epoches):

            print "epoch: " + str(epoch)
            # get a random sequence for this epoch
            sequence = generate_shuffled_array(0, total_data_num, shuffle=True)
            batch_num = int(total_data_num / batch_size)
            # start batches
            for batch_seq in range(batch_num):
                print "batch:" + str(batch_seq)
                # get data for this batch
                start_position = batch_seq * batch_size
                end_position = (batch_seq + 1) * batch_size
                batch_data = get_bacth(sequence[start_position:end_position], data_mat)

                sess.run(train_step, feed_dict={x_: batch_data})  # training

            print "epoch: " + str(epoch)
            print('loss=%s' % sess.run(loss, feed_dict={x_: batch_data}))
            if (epoch+1) % save_every_n_epoch == 0:
                saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/'+'simulation_autoencoder_'+str(epoch)+'.ckpt')


