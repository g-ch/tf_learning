import tensorflow as tf
import numpy as np
import sys
import csv
import gc
import cv2
import matplotlib.pyplot as plt

input_dimension_x = 256
input_dimension_y = 192
input_channel = 2

batch_size = 20
learning_rate = 1e-4
total_epoches = 1000
save_every_n_epoch = 100
times_per_file = 1

model_save_path = "/home/ubuntu/chg_workspace/depth_semantic/model/encoder/01/model/"
image_save_path = "/home/ubuntu/chg_workspace/depth_semantic/model/encoder/01/plots/"

file_path_list_depth_images = [
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/01/depth_data_2019_03_11_11:31:50.csv",
    # "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/02/depth_data_2019_03_11_10:29:56.csv",
    # "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/03/depth_data_2019_03_11_11:00:27.csv",
    # "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/04/depth_data_2019_03_11_10:46:19.csv",
    # "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/05/depth_data_2019_03_11_10:40:38.csv",
    # "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/06/depth_data_2019_03_11_11:12:33.csv"
]

file_path_list_semantic_images = [
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/01/semantics_2019_03_11_11:31:50.csv",
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/02/semantics_2019_03_11_10:29:56.csv",
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/03/semantics_2019_03_11_11:00:27.csv",
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/04/semantics_2019_03_11_10:46:19.csv",
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/05/semantics_2019_03_11_10:40:38.csv",
    "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/long_good/06/semantics_2019_03_11_11:12:33.csv"
]


img_wid = input_dimension_x
img_height = input_dimension_y

encoder_para = {
    "kernel1": 5,
    "stride1": 2,
    "channel1": 32,
    "pool1": 2,
    "kernel2": 3,
    "stride2": 2,
    "channel2": 64,
    "kernel3": 3,
    "stride3": 2,
    "channel3": 128,
    "kernel4": 3,
    "stride4": 2,
    "channel4": 256,
    "out_dia": 12288
}


def conv2d_relu(x, kernel_shape, bias_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def deconv2d(x, kernel_shape, output_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.conv2d_transpose(x, filter=weights, output_shape=output_shape, strides=strides)


def max_pool(x, kernel_shape, strides):
    return tf.nn.max_pool(x, ksize=kernel_shape, strides=strides, padding='SAME')


def encoder(x):
    print "building encoder.."
    k1 = encoder_para["kernel1"]
    s1 = encoder_para["stride1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"]

    k2 = encoder_para["kernel2"]
    s2 = encoder_para["stride2"]
    d2 = encoder_para["channel2"]

    k3 = encoder_para["kernel3"]
    s3 = encoder_para["stride3"]
    d3 = encoder_para["channel3"]

    k4 = encoder_para["kernel4"]
    s4 = encoder_para["stride4"]
    d4 = encoder_para["channel4"]

    print "building encoder"
    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv2d_relu(x, [k1, k1, input_channel, d1], [d1], [1, s1, s1, 1])
            print "conv1 ", conv1
        with tf.variable_scope("conv1_1"):
            conv1_1 = conv2d_relu(conv1, [k1, k1, d1, d1], [d1], [1, 1, 1, 1])

        with tf.variable_scope("pool1"):
            max_pool1 = max_pool(conv1_1, [1, p1, p1, 1], [1, p1, p1, 1])

        with tf.variable_scope("conv2"):
            conv2 = conv2d_relu(max_pool1, [k2, k2, d1, d2], [d2], [1, s2, s2, 1])
        with tf.variable_scope("conv2_1"):
            conv2_1 = conv2d_relu(conv2, [k2, k2, d2, d2], [d2], [1, 1, 1, 1])

        with tf.variable_scope("conv3"):
            conv3 = conv2d_relu(conv2_1, [k3, k3, d2, d3], [d3], [1, s3, s3, 1])
        with tf.variable_scope("conv3_1"):
            conv3_1 = conv2d_relu(conv3, [k3, k3, d3, d3], [d3], [1, 1, 1, 1])

        with tf.variable_scope("conv4"):
            conv4 = conv2d_relu(conv3_1, [k4, k4, d3, d4], [d4], [1, s4, s4, 1])
        with tf.variable_scope("conv4_1"):
            conv4_1 = conv2d_relu(conv4, [k4, k4, d4, d4], [d4], [1, 1, 1, 1])

            return conv4_1


def decoder(x, batch_size):
    k1 = encoder_para["kernel1"]
    s1 = encoder_para["stride1"]
    d1 = encoder_para["channel1"]
    p1 = encoder_para["pool1"]

    k2 = encoder_para["kernel2"]
    s2 = encoder_para["stride2"]
    d2 = encoder_para["channel2"]

    k3 = encoder_para["kernel3"]
    s3 = encoder_para["stride3"]
    d3 = encoder_para["channel3"]

    k4 = encoder_para["kernel4"]
    s4 = encoder_para["stride4"]
    d4 = encoder_para["channel4"]

    size_0 = [batch_size, 192, 256, d1]
    size_1 = [batch_size, 96, 128, d1]
    size_2 = [batch_size, 48, 64, d2]
    size_3 = [batch_size, 24, 32, d3]
    size_4 = [batch_size, 12, 16, d4]
    print "building decoder"

    # Use conv to decrease kernel number. Use deconv to enlarge map

    with tf.variable_scope("decoder"):
        with tf.variable_scope("conv0"):  # Middle layer, change nothing
            conv0 = conv2d_relu(x, [k4, k4, d4, d4], [d4], [1, 1, 1, 1])
            print "conv0 ", conv0

        with tf.variable_scope("deconv0"):
            deconv0 = deconv2d(conv0, [s4, s4, d4, d4], output_shape=size_4, strides=[1, s4, s4, 1])
            print "deconv0", deconv0.get_shape()
        with tf.variable_scope("conv1"):
            conv1 = conv2d_relu(deconv0, [k4, k4, d4, d3], [d3], [1, 1, 1, 1])

        with tf.variable_scope("deconv1"):
            deconv1 = deconv2d(conv1, [s3, s3, d3, d3], output_shape=size_3, strides=[1, s3, s3, 1])
            print "deconv1", deconv1.get_shape()
        with tf.variable_scope("conv2"):
            conv2 = conv2d_relu(deconv1, [k4, k4, d3, d2], [d2], [1, 1, 1, 1])

        with tf.variable_scope("deconv2"):
            deconv2 = deconv2d(conv2, [s2, s2, d2, d2], output_shape=size_2, strides=[1, s2, s2, 1])
            print "deconv2", deconv2.get_shape()
        with tf.variable_scope("conv3"):
            conv3 = conv2d_relu(deconv2, [k3, k3, d2, d1], [d1], [1, 1, 1, 1])

        with tf.variable_scope("deconv3"):
            deconv3 = deconv2d(conv3, [p1, p1, d1, d1], output_shape=size_1, strides=[1, p1, p1, 1])
            print "deconv3", deconv3.get_shape()
        with tf.variable_scope("conv4"):
            conv4 = conv2d_relu(deconv3, [k2, k2, d1, d1], [d1], [1, 1, 1, 1])

        with tf.variable_scope("deconv4"):
            deconv4 = deconv2d(conv4, [s1, s1, d1, d1], output_shape=size_0, strides=[1, s1, s1, 1])
            print "deconv3", deconv4.get_shape()
        with tf.variable_scope("conv5"):
            conv5 = conv2d_relu(deconv4, [k1, k1, d1, input_channel], [input_channel], [1, 1, 1, 1])

            return conv5


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


def read_img_two_channels(filename_img1, filename_img2, house, seq):  # filename_img1 & filename_img2 must have the same length
    maxInt = sys.maxsize
    decrement = True

    img1 = open(filename_img1, "r")
    img_num = len(img1.readlines())
    img1.close()

    data_img = np.zeros([img_num, img_height, img_wid, 2])

    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            print "begin read img data.."
            csv.field_size_limit(maxInt)

            with open(filename_img1, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_height):
                        for j in range(img_wid):
                            data_img[i_row, i, j, 0] = row[i * img_wid + j]
                    i_row = i_row + 1
                # list_result.append(data)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

        decrement = False
        try:
            print "begin read img data.."
            csv.field_size_limit(maxInt)

            with open(filename_img2, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_height):
                        for j in range(img_wid):
                            data_img[i_row, i, j, 1] = row[i * img_wid + j]
                    i_row = i_row + 1
                # list_result.append(data)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

        house[seq] = data_img


def tf_training(data_house, file_num):
    '''Training'''
    print "building network"
    dia_x = input_dimension_x
    dia_y = input_dimension_y
    dia_channel = input_channel
    x_ = tf.placeholder("float", shape=[None, dia_y, dia_x, dia_channel])

    encode_vector = encoder(x_)

    print "encode_vector: ", encode_vector.get_shape()
    decode_result = decoder(encode_vector, batch_size)

    loss = tf.reduce_mean(tf.square(x_ - decode_result))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print "Start training"

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epoches):
            print "epoch: " + str(epoch)

            for file_seq in range(file_num):

                data_mat = data_house[file_seq]

                # get a random sequence for this file
                for times in range(times_per_file):
                    sequence = generate_shuffled_array(0, data_mat.shape[0], shuffle=True)
                    batch_num = int(data_mat.shape[0] / batch_size)
                    # start batches
                    for batch_seq in range(batch_num):
                        # print "batch:" + str(batch_seq)
                        # get data for this batch
                        start_position = batch_seq * batch_size
                        end_position = (batch_seq + 1) * batch_size
                        batch_data = get_bacth(sequence[start_position:end_position], data_mat)

                        sess.run(train_step, feed_dict={x_: batch_data})  # training

            print "epoch: " + str(epoch)
            print('loss=%s' % sess.run(loss, feed_dict={x_: batch_data}))

            if epoch % 10 == 0:
                # decode_img = sess.run(decode_result, feed_dict={x_: batch_data})
                # save_name = image_save_path + "epoch_" + str(epoch) + ".png"
                # img_to_save = decode_img[0, :, :, :]
                # cv2.imwrite(save_name, img_to_save)
                # opencv image save here

                decode_img = sess.run(decode_result, feed_dict={x_: batch_data})
                save_name = image_save_path + "epoch_" + str(epoch) + ".png"
                save_name_ori = image_save_path + "epoch_" + str(epoch) + "_ori.png"

                img_to_save = np.zeros([img_height, img_wid, 3], np.uint8)
                img_to_save[:, :, 0:2] = np.uint8(decode_img[0, :, :, 0:2])

                img_ori_to_save = np.zeros([img_height, img_wid, 3], np.uint8)
                img_ori_to_save[:, :, 0:2] = batch_data[0, :, :, 0:2]

                plt.imsave(save_name_ori, img_ori_to_save)
                plt.imsave(save_name, img_to_save)

            if epoch % save_every_n_epoch == 0:
                saver.save(sess,
                           model_save_path + "simulation_autoencoder_" + str(epoch) + ".ckpt")


if __name__ == '__main__':

    '''Data reading'''
    print "Reading data..."

    file_num = len(file_path_list_depth_images)
    data_house = [0 for i in range(file_num)]

    for file_seq in range(file_num):
        print "reading file " + str(file_seq)
        read_img_two_channels(file_path_list_depth_images[file_seq], file_path_list_semantic_images[file_seq], data_house, file_seq)

    tf_training(data_house, file_num)





