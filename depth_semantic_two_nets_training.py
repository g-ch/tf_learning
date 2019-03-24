import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import time
import gc
import file_walker
import os

''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
learning_rate = 1e-4
regularization_para = 1e-7  # might have problem in rgb training, too many parameters
epoch_num = 500
save_every_n_epoch = 50
training_times_simple_epoch = 2
if_train_encoder = True
if_continue_train = False
if_regularization = True

# model_save_path = "/cluster/home/it_stu45/model/cnn_nornn/01/model/"
# image_save_path = "/cluster/home/it_stu45/model/cnn_nornn/01/plot/"
model_save_path = "/home/ubuntu/chg_workspace/depth_semantic/model/cnn_nornn/01/model/"
image_save_path = "/home/ubuntu/chg_workspace/depth_semantic/model/cnn_nornn/01/plot/"

encoder_model = "/home/ubuntu/chg_workspace/depth/model/encoder/01/model/simulation_autoencoder_500.ckpt"
last_model = ""

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_x": 256,
    "input1_dim_y": 192,
    "input1_dim_channel": 1,
    "input2_dim": 4  # commands
}

input_semantic_paras = {
    "input_dim_x": 256,
    "input_dim_y": 192,
    "input_dim_channel": 1,
}

commands_compose_each = 1  # Should be "input3_dim": 4  / 4

input_dimension_x = input_paras["input1_dim_x"]
input_dimension_y = input_paras["input1_dim_y"]
input_channel = input_paras["input1_dim_channel"]

img_wid = input_dimension_x
img_height = input_dimension_y
img_channel = input_channel

''' Parameters for csv files '''

states_num_one_line = 17
labels_num_one_line = 4

#training_file_path = "/cluster/home/it_stu45/dataset"
training_file_path = "/home/ubuntu/chg_workspace/data/new_map_with_deepth_img/deepth_rgb_seemantics/gazebo_rate_092/yhz/short/32"

''' Parameters for Computer'''
gpu_num = 2

''' Parameters for concat fully layers'''
fully_paras = {
    "raw_batch_size": 20,
    "input_len": 576,
    "layer1_len": 256,
    "layer2_len": 64,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 512,  # should be the same as encoder out dim
    "dim2": 32,  # dim1 + dim2 + dim3 should be input_len of the rnn, for line vector
    "dim3": 32   # for semantic info
}

''' Parameters for CNN encoder'''
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

''' Parameters for semantic part'''
semantic_para = {
    "kernel1": 5,
    "stride1": 2,
    "channel1": 8,
    "pool1": 2,
    "kernel2": 3,
    "stride2": 2,
    "channel2": 16,
    "pool2": 2,
    "kernel3": 3,
    "stride3": 2,
    "channel3": 32,
    "out_dia": 1536,
    "fully1": 64,
    "fully2": 32
}


def conv2d_relu(x, kernel_shape, bias_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def max_pool(x, kernel_shape, strides):
    return tf.nn.max_pool(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu_layer(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights", [x_diamension, neurals_num],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

    if if_regularization:
        # L2 regularization
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularization_para)(weights))

    biases = tf.get_variable("bias", [neurals_num], initializer=tf.constant_initializer(0.1))
    return tf.nn.relu(tf.matmul(x, weights) + biases)


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


def semantic_networks(x):
    print "building semantic network.."
    k1 = semantic_para["kernel1"]
    s1 = semantic_para["stride1"]
    d1 = semantic_para["channel1"]
    p1 = semantic_para["pool1"]

    k2 = semantic_para["kernel2"]
    s2 = semantic_para["stride2"]
    d2 = semantic_para["channel2"]
    p2 = semantic_para["pool2"]

    k3 = semantic_para["kernel3"]
    s3 = semantic_para["stride3"]
    d3 = semantic_para["channel3"]

    out_dia = semantic_para["out_dia"]
    f1 = semantic_para["fully1"]
    f2 = semantic_para["fully2"]

    with tf.variable_scope("semantic"):
        with tf.variable_scope("conv1"):
            conv1 = conv2d_relu(x, [k1, k1, input_semantic_paras["input_dim_channel"], d1], [d1], [1, s1, s1, 1])

        with tf.variable_scope("pool1"):
            max_pool1 = max_pool(conv1, [1, p1, p1, 1], [1, p1, p1, 1])

        with tf.variable_scope("conv2"):
            conv2 = conv2d_relu(max_pool1, [k2, k2, d1, d2], [d2], [1, s2, s2, 1])

        with tf.variable_scope("pool2"):
            max_pool2 = max_pool(conv2, [1, p2, p2, 1], [1, p2, p2, 1])

        with tf.variable_scope("conv3"):
            conv3 = conv2d_relu(max_pool2, [k3, k3, d2, d3], [d3], [1, s3, s3, 1])

        with tf.variable_scope("fully1"):
            vector_flat = tf.reshape(conv3, [-1, out_dia])
            fully1 = relu_layer(vector_flat, out_dia, f1)

        with tf.variable_scope("fully2"):
            fully2 = relu_layer(fully1, f1, f2)

            return fully2


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


def read_threading(filename_img1, filename_img2, filename_state, filename_label, file_seq, data_house):
    """
    Read data thread function.
    :param filename_pcl:  pcl filename
    :param filename_state: state filename
    :param filename_label: label filename
    :param data_read_flags: flags to find a empty place
    :param house: house to store data, composed of [[[pcl], [state1], [state2], [label]],   [],   [],   []...]
    :return:
    """
    print "Start reading..."
    ''' Read pcl data first '''
    clouds = open(filename_img1, "r")
    img_num = len(clouds.readlines())
    clouds.close()
    data_img1 = np.zeros([img_num, img_height, img_wid, img_channel])
    data_img2 = np.zeros([img_num, img_height, img_wid, input_semantic_paras["input_dim_channel"]])
    read_img_two_channels(data_img1, data_img2,  filename_img1, filename_img2)
    print "rgb data get! img_num = " + str(img_num)

    # Just to make sure the data is read correctly
    # compare_draw_3d_to_2d(data_pcl[10, :, :, :, 0], data_pcl[10, :, :, :, 0], 0, 1, 2, 12, 1)

    ''' Read state data '''
    data_states = np.zeros([img_num, states_num_one_line])
    # print "state name", filename_state
    read_others(data_states, filename_state, states_num_one_line)
    # print "state data get!"

    ''' Read label data '''
    data_labels = np.zeros([img_num, labels_num_one_line])
    read_others(data_labels, filename_label, labels_num_one_line)
    # print "label data get!"

    ''' Get useful states and labels '''
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
    # print labels_ref
    labels_ref[:, 1] = (0.8 * labels_ref[:, 1] + np.ones(img_num)) / 2.0  # !!!!!
    # print labels_ref

    ''' Store data to house '''
    data_house[file_seq] = [data_img1, data_img2, commands_input, labels_ref]


def read_img_two_channels(data_img1, data_img2, filename_img1, filename_img2):  # filename_img1 & filename_img2 must have the same length
    maxInt = sys.maxsize
    decrement = True

    # data_img = np.zeros([img_num, img_height, img_wid, 2])

    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            print "begin read depth img data.."
            csv.field_size_limit(maxInt)

            with open(filename_img1, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_height):
                        for j in range(img_wid):
                            data_img1[i_row, i, j, 0] = row[i * img_wid + j]
                    i_row = i_row + 1
                # list_result.append(data)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

        decrement = False
        try:
            print "begin read semantic img data.."
            csv.field_size_limit(maxInt)

            with open(filename_img2, mode='r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                i_row = 0
                for row in csv_reader:
                    for i in range(img_height):
                        for j in range(img_wid):
                            data_img2[i_row, i, j, 0] = row[i * img_wid + j]
                    i_row = i_row + 1
                # list_result.append(data)
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


def tf_training(data_house, file_num):
    """
    Main training function
    :param data_read_flags: flag to find stored data
    :param data_house: where the data stores
    :param file_num: total file number of input csvs
    :return:
    """
    ''' Calculate batch size '''
    batch_size_one_gpu = fully_paras["raw_batch_size"]
    batch_size = batch_size_one_gpu * gpu_num

    ''' Graph building '''
    print "Building graph!"
    with tf.device("/cpu:0"):
        global_step = tf.train.get_or_create_global_step()
        tower_grads = []
        depth_data = tf.placeholder("float", name="rgb_data", shape=[None, input_dimension_y, input_dimension_x,
                                    input_channel])
        semantic_data = tf.placeholder("float", name="semantic_data", shape=[None, input_semantic_paras["input_dim_y"],
                                                                             input_semantic_paras["input_dim_x"],
                                                                             input_semantic_paras["input_dim_channel"]])

        line_data_2 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input2_dim"]])  # commands
        reference = tf.placeholder("float", name="reference", shape=[None, fully_paras["output_len"]])

        # Optimizer
        train_step = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_seq in range(gpu_num):
                with tf.device("/gpu:%d" % gpu_seq):
                    # Set data for each gpu
                    depth_data_this_gpu = depth_data[gpu_seq * batch_size_one_gpu:
                                                     (gpu_seq + 1) * batch_size_one_gpu, :, :, :]
                    semantic_data_this_gpu = semantic_data[gpu_seq * batch_size_one_gpu:
                                                           (gpu_seq + 1) * batch_size_one_gpu, :, :, :]
                    line_data_2_this_gpu = line_data_2[gpu_seq * batch_size_one_gpu:
                                                       (gpu_seq + 1) * batch_size_one_gpu, :]
                    reference_this_gpu = reference[gpu_seq * batch_size_one_gpu:(gpu_seq + 1) * batch_size_one_gpu, :]

                    # Semantic data
                    semantic_vector_flat = semantic_networks(semantic_data_this_gpu)

                    # depth
                    encode_vector = encoder(depth_data_this_gpu)
                    print "encoder built"
                    # To flat vector
                    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["out_dia"]])
                    # Dropout 1
                    encode_vector_flat = tf.layers.dropout(encode_vector_flat, rate=0.5, training=True)

                    # Add a fully connected layer for map
                    with tf.variable_scope("relu_encoder_1"):
                        map_data_line_0 = relu_layer(encode_vector_flat, encoder_para["out_dia"], concat_paras["dim1"])
                    with tf.variable_scope("relu_encoder_2"):
                        map_data_line = relu_layer(map_data_line_0, concat_paras["dim1"], concat_paras["dim1"])

                    # Add a fully connected layer for commands
                    with tf.variable_scope("relu_commands_1"):
                        commands_data_line_0 = relu_layer(line_data_2_this_gpu, input_paras["input2_dim"],
                                                          concat_paras["dim2"])
                    with tf.variable_scope("relu_commands_2"):
                        commands_data_line = relu_layer(commands_data_line_0, concat_paras["dim2"],
                                                        concat_paras["dim2"])

                    # Concat, Note: dimension parameter should be 1, considering batch size
                    concat_vector = tf.concat([map_data_line, commands_data_line, semantic_vector_flat], 1)
                    print "concat complete"

                    # Add a fully connected layer for all input
                    with tf.variable_scope("relu_all_1"):
                        relu_data_all = relu_layer(concat_vector, fully_paras["input_len"],
                                                   fully_paras["input_len"])
                    # Dropout 2
                    relu_data_droped = tf.layers.dropout(relu_data_all, rate=0.5, training=True)

                    with tf.variable_scope("relu_all_2"):
                        relu_data_all_2 = relu_layer(relu_data_droped, fully_paras["input_len"],
                                                     fully_paras["layer1_len"])

                    with tf.variable_scope("relu_all_3"):
                        relu_data_all_3 = relu_layer(relu_data_all_2, fully_paras["layer1_len"],
                                                     fully_paras["layer2_len"])

                    with tf.variable_scope("relu_all_4"):
                        result_this_gpu = relu_layer(relu_data_all_3, fully_paras["layer2_len"],
                                                     fully_paras["output_len"])

                    print "graph built!"

                    tf.get_variable_scope().reuse_variables()

                    if if_regularization:
                        ses_loss = tf.reduce_mean(tf.square(reference_this_gpu - result_this_gpu))
                        tf.add_to_collection("losses", ses_loss)
                        loss = tf.add_n(tf.get_collection("losses"))
                    else:
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

            # start epochs
            for epoch in range(epoch_num):
                print "epoch: " + str(epoch)
                t0 = time.time()

                seq_array = generate_shuffled_array(0, file_num)
                ''' waiting for data '''
                for file_seq in range(file_num):
                    read_seq = seq_array[file_seq]

                    data_mat_depth = data_house[read_seq][0]
                    data_mat_semantic = data_house[read_seq][1]
                    data_mat_command = data_house[read_seq][2]
                    data_mat_label = data_house[read_seq][3]
                    data_num = data_mat_depth.shape[0]

                    batch_num = int(data_num / batch_size)

                    for training_time_this_file in range(training_times_simple_epoch):
                        # get a random sequence for this file
                        sequence = generate_shuffled_array(0, data_num, shuffle=True)

                        # start batches
                        for batch_seq in range(batch_num):
                            print "batch" + str(batch_seq)
                            # get data for this batch
                            start_position = batch_seq * batch_size
                            end_position = (batch_seq + 1) * batch_size
                            data_depth_batch = get_batch(sequence[start_position:end_position], data_mat_depth)
                            data_semantic_batch = get_batch(sequence[start_position:end_position], data_mat_semantic)
                            data_command_batch = get_batch(sequence[start_position:end_position], data_mat_command)
                            label_batch = get_batch(sequence[start_position:end_position], data_mat_label)

                            # label_to_draw = np.reshape(label_batch[:, 0], [batch_size])
                            # draw_plots(np.arange(0, batch_size), label_to_draw)

                            # train
                            sess.run(train_op, feed_dict={depth_data: data_depth_batch, semantic_data:data_semantic_batch,
                                                          line_data_2: data_command_batch, reference: label_batch})

                            del data_depth_batch
                            del data_semantic_batch
                            del data_command_batch
                            del label_batch

                    del data_mat_depth
                    del data_mat_semantic
                    del data_mat_command
                    del data_mat_label
                    del data_num

                print "Epoch " + str(epoch) + " finished!  Time: " + str(time.time() - t0)

                if epoch % save_every_n_epoch == 0:
                    # save
                    saver.save(sess, model_save_path + "simulation_cnn_rnn" + str(epoch) + ".ckpt")


if __name__ == '__main__':
    ''' Search for training data in the training folder '''
    scan = file_walker.ScanFile(training_file_path)
    files = scan.scan_files()

    file_path_depth = []
    file_path_semantic = []
    file_path_states = []
    file_path_labels = []

    file_type = '.csv'
    for file in files:
        if os.path.splitext(file)[1] == file_type:
            if os.path.splitext(file)[0].split('/')[-1].split('_')[0] == 'depth':
                file_path_depth.append(file)
                file_path_states.append(file.replace('depth', 'uav'))
                file_path_labels.append(file.replace('depth', 'label'))
                file_path_semantic.append(file.replace('depth_data', 'semantics'))

    print "Found " + str(len(file_path_depth)) + " files to train!!!"

    files_num = len(file_path_depth)
    data_house = [0 for i in range(files_num)]

    # Data Reading
    for seq in range(files_num):
        # pool.apply_async(test)
        filename_depth_this = file_path_depth[seq]
        filename_states_this = file_path_states[seq]
        filename_labels_this = file_path_labels[seq]
        filename_semantics_this = file_path_semantic[seq]

        read_threading(filename_depth_this, filename_semantics_this, filename_states_this, filename_labels_this, seq, data_house)

    tf_training(data_house, files_num)

