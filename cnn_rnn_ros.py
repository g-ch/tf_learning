import tensorflow as tf
import math
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


''' Parameters for training '''
''' Batch size defined in Parameters for RNN '''
test_data_num = 400
input_side_dimension = 64

img_width = input_side_dimension
img_height = input_side_dimension

model_path = "/home/ubuntu/chg_workspace/3dcnn/model/model_cnn_rnn_timestep5/simulation_cnn_rnn200.ckpt"
compose_num = [200, 28, 28]

''' Parameters for RNN'''
rnn_paras = {
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

''' Parameters for ros node '''
new_msg_received = False

position_odom_x = -1
position_odom_y = -1
position_odom_z = -1
yaw_delt = 0.0
velocity_odom_linear = 0.0
velocity_odom_angular = 0.0

pcl_arr = np.ones(dtype=np.float32, shape=[1, img_width, img_width, img_height, 1])


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


class Networkerror(RuntimeError):
    """
    Error print
    """
    def __init__(self, arg):
        self.args = arg


def refillPclArr(arr, point_x, point_y, point_z, intensity, odom_x, odom_y, odom_z):
    x_tmp = int((point_x - odom_x) * 5 + 0.5)
    y_tmp = int((point_y - odom_y) * 5 + 0.5)
    z_tmp = int((point_z - odom_z) * 5 + 0.5)

    if abs(x_tmp) < img_width/2 and abs(y_tmp) < img_width/2 and abs(z_tmp) < img_height/2:
        x_tmp = int(x_tmp + img_width / 2)
        y_tmp = int(y_tmp + img_width / 2)
        z_tmp = int(z_tmp + img_height / 2)
        arr[0, x_tmp, y_tmp, z_tmp, 0] = intensity

    return arr


def callBackPCL(point):
    global pcl_arr
    pcl_arr = np.ones(dtype=np.float32, shape=[1, img_width, img_width, img_height, 1])
    for p in pc2.read_points(point, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        pcl_arr = refillPclArr(pcl_arr, p[0], p[1], p[2], p[3]/7.0, position_odom_x, position_odom_y, position_odom_z)
    global new_msg_received
    new_msg_received = True


def callBackDeltYaw(data):
    global yaw_delt
    yaw_delt = data.data


def callBackOdom(data):
    global position_odom_x, position_odom_y, velocity_odom_angular, velocity_odom_linear
    position_odom_x, position_odom_y, position_odom_z = \
        data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z
    velocity_odom_linear, velocity_odom_angular = data.twist.twist.linear.x, data.twist.twist.angular.z


if __name__ == '__main__':

    rospy.init_node('predict', anonymous=True)
    rospy.Subscriber('/ring_buffer/cloud_semantic', PointCloud2, callBackPCL)
    rospy.Subscriber("/radar/delt_yaw", Float64, callBackDeltYaw)
    rospy.Subscriber("/odom", Odometry, callBackOdom)
    cmd_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)
    move_cmd = Twist()

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

    cube_dim = input_side_dimension
    rate = rospy.Rate(100)  # 100hz

    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
    config.gpu_options.allow_growth = True  # only 300M memory

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, model_path)
        state_data_give = np.zeros([1, rnn_paras["state_len"]])

        counter = 0
        while not rospy.is_shutdown():
            if new_msg_received:
                data2_yaw = np.ones([1, compose_num[0]]) * yaw_delt
                data2_vl = np.ones([1, compose_num[1]]) * velocity_odom_linear
                data2_va = np.ones([1, compose_num[2]]) * velocity_odom_angular

                data2_to_feed = np.concatenate([data2_yaw, data2_vl, data2_va], axis=1)

                results = sess.run(result, feed_dict={cube_data: pcl_arr, line_data: data2_to_feed,
                                                      state_data: state_data_give})
                move_cmd.linear.x = -results[0, 0] * 0.4
                move_cmd.angular.z = results[0, 1] * 1.5

                if counter > 40:
                    cmd_pub.publish(move_cmd)

                counter = counter + 1

                print results
                state_data_give = sess.run(state_returned,
                                           feed_dict={cube_data: pcl_arr, line_data: data2_to_feed,
                                                      state_data: state_data_give})
                new_msg_received = False
            rate.sleep()





