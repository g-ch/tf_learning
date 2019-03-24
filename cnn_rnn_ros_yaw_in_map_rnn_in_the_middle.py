import tensorflow as tf
import math
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

commands_compose_each = 1  # Should be "input3_dim": 8  / 4
if_train_encoder = False

model_path = "/home/ubuntu/chg_workspace/3dcnn_yaw_in_map/model/cnn_rnn/05_rnn_in_middle/model/simulation_cnn_rnn320.ckpt"

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_xy": 64,  # point cloud
    "input1_dim_z": 24,  # point cloud
    "input2_dim": 4  # commands
}
input_dimension_xy = input_paras["input1_dim_xy"]
input_dimension_z = input_paras["input1_dim_z"]
img_wid = input_dimension_xy
img_height = input_dimension_z
img_height_uplimit = 20
img_height_downlimit = 4

''' Parameters for RNN'''
rnn_paras = {
    "raw_batch_size": 15,
    "time_step": 8,
    "state_len": 256,
    "input_len": 1088,
    "output_len": 256
}
final_out_dia = 2

''' Parameters for concat values'''
concat_paras = {
    "dim1": 1024,  # should be the same as encoder out dim
    "dim2": 64  # dim1 + dim2 + dim3 should be input_len of the rnn, for line vector
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

''' Parameters for ros node '''
new_msg_received = False

position_odom_x = 0.0
position_odom_y = 0.0
position_odom_z = 0.0
yaw_delt = 0.0
yaw_current = 0.0
yaw_current_x = 0.0
yaw_current_y = 0.0
velocity_odom_linear = 0.0
velocity_odom_angular = 0.0
yaw_forward = 0.0
yaw_backward = 0.0
yaw_leftward = 0.0
yaw_rightward = 0.0

pcl_arr = np.ones(dtype=np.float32, shape=[1, img_wid, img_wid, img_height, 1])

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
        state = tf.nn.relu(tf.matmul(state_last, u) + tf.matmul(x, w))  # hidden layer activate function
        return tf.nn.relu(tf.matmul(state, v) + b), state  # output layer activate function


def myrnn_training(x, input_len, output_len, raw_batch_size, time_step, state_len):
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
            state = tf.nn.relu(tf.matmul(state, u) + tf.matmul(x_temp, w))  # hidden layer activate function

        return tf.nn.relu(tf.matmul(state, v) + b)  # output layer activate function


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


class Networkerror(RuntimeError):
    """
    Error print
    """
    def __init__(self, arg):
        self.args = arg


def refillPclArr(arr, point_x, point_y, point_z, intensity, odom_x, odom_y, odom_z):
    resolu = 0.2
    global yaw_current
    yaw = yaw_current * 3.15
    x_origin = point_x - odom_x
    y_origin = point_y - odom_y

    time_now = time.time()
    x_rotated, y_rotated = rotate(x_origin, y_origin, yaw)
    time_elapsed = time.time() - time_now
    # print("Time used for rotation:  " + str(time_elapsed))

    x_tmp = int(x_rotated / resolu + 0.5)
    y_tmp = int(y_rotated / resolu + 0.5)
    z_tmp = int((point_z - odom_z) / resolu + 0.5)

    if abs(x_tmp) < img_wid/2 and abs(y_tmp) < img_wid/2 and z_tmp < img_height_uplimit and z_tmp > -img_height_downlimit:
        x_index = int(x_tmp + img_wid / 2)
        y_index = int(y_tmp + img_wid / 2)
        z_index = int(z_tmp + img_height_downlimit)
        arr[0, x_index, y_index, z_index, 0] = intensity

    return arr


def rotate(x_origin, y_origin, yaw_angle):
    x_rotated = x_origin * np.cos(yaw_angle) + y_origin * np.sin(yaw_angle)
    y_rotated = y_origin * np.cos(yaw_angle) - x_origin * np.sin(yaw_angle)
    return x_rotated, y_rotated


def callBackPCL(point):
    global pcl_arr
    # unknown: zeros
    pcl_arr = np.zeros(dtype=np.float32, shape=[1, img_wid, img_wid, img_height, 1])
    for p in pc2.read_points(point, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        intensity = 0.0
        if p[3] == 1.0:
            intensity = p[3] * 0.5
        else:
            intensity = p[3]
        pcl_arr = refillPclArr(pcl_arr, p[0], p[1], p[2], intensity/7.0, position_odom_x, position_odom_y, position_odom_z)
    global new_msg_received
    new_msg_received = True


def callBackDeltYaw(data):
    global yaw_delt
    yaw_delt = data.data

    global yaw_forward
    global yaw_backward
    global yaw_leftward
    global yaw_rightward

    if -3.15 / 4.0 < yaw_delt < 3.15 / 4.0:
        yaw_forward = 1.0
        yaw_backward = 0.0
        yaw_leftward = 0.0
        yaw_rightward = 0.0
    elif 3.15 / 4.0 * 3.0 < yaw_delt or yaw_delt < -3.15 / 4.0 * 3.0:
        yaw_forward = 0.0
        yaw_backward = 1.0
        yaw_leftward = 0.0
        yaw_rightward = 0.0
    elif 3.15 / 4.0 < yaw_delt < 3.15 / 4.0 * 3.0:
        yaw_forward = 0.0
        yaw_backward = 0.0
        yaw_leftward = 1.0
        yaw_rightward = 0.0
    else:
        yaw_forward = 0.0
        yaw_backward = 0.0
        yaw_leftward = 0.0
        yaw_rightward = 1.0

    yaw_delt = data.data / 3.15


def callBackCurrentYaw(data):
    global yaw_current, yaw_current_x, yaw_current_y
    yaw_current = data.data / 3.15
    yaw_current_x = math.cos(data.data)
    yaw_current_y = math.sin(data.data)


def callBackOdom(data):
    global position_odom_x, position_odom_y, position_odom_z, velocity_odom_angular, velocity_odom_linear
    position_odom_x, position_odom_y, position_odom_z = \
        data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z
    # input last velocity for rnn # !!max velocity=0.8, max angular_velocity=1.0
    velocity_odom_linear, velocity_odom_angular = data.twist.twist.linear.x / 0.8, data.twist.twist.angular.z


# draw by axis z direction
def compare_save_3d_to_2d(data1, data2, min_val, max_val, rows, cols, step):
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
    plt.plot(x, y)

    plt.title("matplotlib")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    rospy.init_node('predict', anonymous=True)
    rospy.Subscriber('/ring_buffer/cloud_semantic', PointCloud2, callBackPCL)
    rospy.Subscriber("/radar/delt_yaw", Float64, callBackDeltYaw)
    rospy.Subscriber("/radar/current_yaw", Float64, callBackCurrentYaw)
    rospy.Subscriber("/odom", Odometry, callBackOdom)
    cmd_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)
    move_cmd = Twist()

    ''' Graph building '''
    cube_data = tf.placeholder("float", name="cube_data", shape=[None, input_dimension_xy, input_dimension_xy,
                                                                 input_dimension_z, 1])
    line_data_2 = tf.placeholder("float", name="line_data_2", shape=[None, input_paras["input2_dim"]])  # commands
    state_data = tf.placeholder("float", name="state_data", shape=[1, rnn_paras["state_len"]])

    # 3D CNN
    encode_vector = encoder(cube_data)
    # To flat vector
    encode_vector_flat = tf.reshape(encode_vector, [-1, encoder_para["out_dia"]])

    # Add a fully connected layer for map
    with tf.variable_scope("relu_encoder_1"):
        map_data_line_0 = relu_layer(encode_vector_flat, encoder_para["out_dia"], concat_paras["dim1"])
    with tf.variable_scope("relu_encoder_2"):
        map_data_line = relu_layer(map_data_line_0, concat_paras["dim1"], concat_paras["dim1"])
    # Add a fully connected layer for commands
    with tf.variable_scope("relu_commands_1"):
        commands_data_line_0 = relu_layer(line_data_2, input_paras["input2_dim"],
                                          concat_paras["dim2"])
    with tf.variable_scope("relu_commands_2"):
        commands_data_line = relu_layer(commands_data_line_0, concat_paras["dim2"],
                                        concat_paras["dim2"])

    # Concat, Note: dimension parameter should be 1, considering batch size
    concat_vector = tf.concat([map_data_line, commands_data_line], 1)

    # Add a fully connected layer for all input before rnn
    with tf.variable_scope("relu_all_1"):
        relu_data_all = relu_layer(concat_vector, rnn_paras["input_len"],
                                   rnn_paras["input_len"])

    # Feed to rnn
    rnn_output, state_returned = myrnn_test(relu_data_all, state_data, rnn_paras["input_len"], rnn_paras["output_len"],
                                        rnn_paras["state_len"])

    with tf.variable_scope("relu_all_2"):
        result = relu_layer(rnn_output, rnn_paras["output_len"], final_out_dia)

    ''' Predicting '''
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['rnn/state'])
    restorer = tf.train.Saver(variables_to_restore)

    rate = rospy.Rate(100)  # 100hz

    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
    config.gpu_options.allow_growth = True  # only 300M memory

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, model_path)
        state_data_give = np.zeros([1, rnn_paras["state_len"]])

        print "parameters restored!"
        global new_msg_received

        while not rospy.is_shutdown():
            if new_msg_received:
                data3_yaw_forward = np.ones([1, commands_compose_each]) * yaw_forward
                data3_yaw_backward = np.ones([1, commands_compose_each]) * yaw_backward
                data3_yaw_leftward = np.ones([1, commands_compose_each]) * yaw_leftward
                data3_yaw_rightward = np.ones([1, commands_compose_each]) * yaw_rightward
                data3_to_feed = np.concatenate([data3_yaw_forward, data3_yaw_backward, data3_yaw_leftward, data3_yaw_rightward], axis=1)

                results = sess.run(result, feed_dict={cube_data: pcl_arr,
                                                      line_data_2: data3_to_feed, state_data: state_data_give})

                state_data_give = sess.run(state_returned,
                                           feed_dict={cube_data: pcl_arr,
                                                      line_data_2: data3_to_feed, state_data: state_data_give})

                # compare_save_3d_to_2d(pcl_arr[0, :, :, :, 0], pcl_arr[0, :, :, :, 0], 0, 1, 4, 12, 1)
                # concat_vector = sess.run(concat_vector, feed_dict={cube_data: pcl_arr, line_data_1: data2_to_feed,
                   #                                   line_data_2: data3_to_feed, state_data: state_data_give})
                # draw_plots(np.arange(0, 576), np.reshape(concat_vector, [576]))

                move_cmd.linear.x = results[0, 0] * 0.7 #1.0
                # # move_cmd.linear.x = 0.0
                move_cmd.angular.z = (2 * results[0, 1] - 1) * 0.88   #0.88

                # if move_cmd.linear.x < 0:
                #     move_cmd.linear.x = 0

                cmd_pub.publish(move_cmd)

                # print yaw_forward, yaw_backward, yaw_leftward, yaw_rightward
                print results

                new_msg_received = False
            rate.sleep()

