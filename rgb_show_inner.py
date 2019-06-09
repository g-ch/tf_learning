import tensorflow as tf
import math
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import cv2
import time
from sensor_msgs.msg import Joy


commands_compose_each = 1  # Should be "input3_dim": 8  / 4

model_path = "/home/gg/intel_nav_ws/src/EAI-E1/scripts/model/rgb/simulation_cnn_rnn200.ckpt"

''' Parameters for input vectors'''
input_paras = {
    "input1_dim_x": 256,
    "input1_dim_y": 192,
    "input1_dim_channel": 3,
    "input2_dim": 4  # commands
}

input_dimension_x = input_paras["input1_dim_x"]
input_dimension_y = input_paras["input1_dim_y"]
input_channel = input_paras["input1_dim_channel"]

img_wid = input_dimension_x
img_height = input_dimension_y
img_channel = input_channel

''' Parameters for concat fully layers'''
fully_paras = {
    "raw_batch_size": 20,
    "input_len": 544,
    "layer1_len": 256,
    "layer2_len": 64,
    "output_len": 2
}

''' Parameters for concat values'''
concat_paras = {
    "dim1": 512,  # should be the same as encoder out dim
    "dim2": 32  # dim1 + dim2 + dim3 should be input_len of the rnn, for line vector
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
emergency_stop = 0.0

direction_msg = Float64MultiArray()

rgb_image = np.zeros([1, img_height, img_wid, img_channel])  # bgr in opencv form
bridge = CvBridge()


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

    biases = tf.get_variable("bias", [neurals_num], initializer=tf.constant_initializer(0.1))
    return tf.nn.relu(tf.matmul(x, weights) + biases)


def encoder(x):
    print ("building encoder..")
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

    print ("building encoder")
    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            conv1 = conv2d_relu(x, [k1, k1, input_channel, d1], [d1], [1, s1, s1, 1])
            print ("conv1 ", conv1)
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

            return conv4_1, conv2_1


class Networkerror(RuntimeError):
    """
    Error print
    """
    def __init__(self, arg):
        self.args = arg


def callBackRGB(img):
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    global rgb_image, new_msg_received
    rgb_image_list = cv2.resize(cv_image, (img_wid, img_height), interpolation=cv2.INTER_AREA)
    # cv2.imshow("rgb", rgb_image_list)
    # cv2.waitKey(5)
    rgb_image = np.array(rgb_image_list).reshape(1, img_height, img_wid, img_channel)
    new_msg_received = True


# def callBackDeltYaw(data):
#     global yaw_delt
#     yaw_delt = data.data

#     global yaw_forward
#     global yaw_backward
#     global yaw_leftward
#     global yaw_rightward

#     if -3.15 / 4.0 < yaw_delt < 3.15 / 4.0:
#         yaw_forward = 1.0
#         yaw_backward = 0.0
#         yaw_leftward = 0.0
#         yaw_rightward = 0.0
#     elif 3.15 / 4.0 * 3.0 < yaw_delt or yaw_delt < -3.15 / 4.0 * 3.0:
#         yaw_forward = 0.0
#         yaw_backward = 1.0
#         yaw_leftward = 0.0
#         yaw_rightward = 0.0
#     elif 3.15 / 4.0 < yaw_delt < 3.15 / 4.0 * 3.0:
#         yaw_forward = 0.0
#         yaw_backward = 0.0
#         yaw_leftward = 1.0
#         yaw_rightward = 0.0
#     else:
#         yaw_forward = 0.0
#         yaw_backward = 0.0
#         yaw_leftward = 0.0
#         yaw_rightward = 1.0

#     yaw_delt = data.data / 3.15


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


def joyCallback(data_joy):
    """
    Use joy command as the desired global direction
    """
    global yaw_forward
    global yaw_backward
    global yaw_leftward
    global yaw_rightward
    global emergency_stop
    issignalchanged = False

    for button in data_joy.buttons:
        if button != 0.0:
            issignalchanged = True
            break
    if not issignalchanged:
        for axe in data_joy.axes:
            if axe != 0.0:
                issignalchanged = True
                break
    if issignalchanged:
        if (data_joy.buttons[8] == 1):
            # stop, corresponding to the 'sahre' button on the joystick
            emergency_stop = 1.0
            yaw_forward = 0.0
            yaw_backward = 0.0
            yaw_leftward = 0.0
            yaw_rightward = 0.0
        else:
            emergency_stop = 0.0
            if (data_joy.axes[7] == 1):
                # forward
                yaw_forward = 1.0
                yaw_backward = 0.0
                yaw_leftward = 0.0
                yaw_rightward = 0.0
            elif (data_joy.axes[7] == -1):
                # backward
                yaw_forward = 0.0
                yaw_backward = 1.0
                yaw_leftward = 0.0
                yaw_rightward = 0.0
            elif (data_joy.axes[6] == 1):
                # left
                yaw_forward = 0.0
                yaw_backward = 0.0
                yaw_leftward = 1.0
                yaw_rightward = 0.0
            elif (data_joy.axes[6] == -1):
                # right
                yaw_forward = 0.0
                yaw_backward = 0.0
                yaw_leftward = 0.0
                yaw_rightward = 1.0
        direction_msg.data = [yaw_forward, yaw_backward, yaw_leftward, yaw_rightward]
        direction_pub.publish(direction_msg)
    else:
        return


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


def merge_images(images, scale):
    size_x = np.shape(images)[1]
    size_y = np.shape(images)[2]
    channels = np.shape(images)[3]
    image_merged = np.zeros([size_x, size_y])

    for k in range(channels):
        image_merged += images[0, :, :, k]

    image_merged = image_merged / channels / scale
    return image_merged


if __name__ == '__main__':

    rospy.init_node('predict', anonymous=True)
    rospy.Subscriber("/kinect2/hd/image_color", Image, callBackRGB)
    # rospy.Subscriber("/radar/delt_yaw", Float64, callBackDeltYaw)
    # rospy.Subscriber("/radar/current_yaw", Float64, callBackCurrentYaw)
    # rospy.Subscriber("/odom", Odometry, callBackOdom)
    rospy.Subscriber("joy", Joy, joyCallback)
    direction_pub = rospy.Publisher("/radar/direction", Float64MultiArray, queue_size=1)
    cmd_pub = rospy.Publisher("smoother_cmd_vel", Twist, queue_size=10)
    move_cmd = Twist()
    status_pub = rospy.Publisher("/robot/status", Float64, queue_size=1)
    status_flag = Float64()
    status_flag.data = 0.0

    ''' Graph building '''
    rgb_data = tf.placeholder("float", name="rgb_data", shape=[None, input_dimension_y, input_dimension_x,
                                                               input_channel])
    line_data_2 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input2_dim"]])  # commands

    # 3D CNN
    encode_vector, image_to_show = encoder(rgb_data)
    print ("encoder built")
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
    print ("concat complete")

    # Add a fully connected layer for all input
    with tf.variable_scope("relu_all_1"):
        relu_data_all = relu_layer(concat_vector, fully_paras["input_len"],
                                   fully_paras["input_len"])

    with tf.variable_scope("relu_all_2"):
        relu_data_all_2 = relu_layer(relu_data_all, fully_paras["input_len"],
                                     fully_paras["layer1_len"])

    with tf.variable_scope("relu_all_3"):
        relu_data_all_3 = relu_layer(relu_data_all_2, fully_paras["layer1_len"],
                                     fully_paras["layer2_len"])

    with tf.variable_scope("relu_all_4"):
        result = relu_layer(relu_data_all_3, fully_paras["layer2_len"], fully_paras["output_len"])

    ''' Predicting '''
    variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)

    rate = rospy.Rate(100)  # 100hz

    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
    config.gpu_options.allow_growth = True  # only 300M memory

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())  # initialze variables
        restorer.restore(sess, model_path)

        value = tf.get_default_graph().get_tensor_by_name("relu_all_1/weights:0")
        value_numpy = value.eval(session=sess)
        np.savetxt('rgb_concat_weights.csv', value_numpy, fmt='%lf', delimiter=",")

        print value_numpy

        print ("parameters restored!")
        global new_msg_received

        while not rospy.is_shutdown():
            if new_msg_received:
                if emergency_stop == 1.0:
                    move_cmd.linear.x = 0.0
                    move_cmd.angular.z = 0.0
                else:
                    data3_yaw_forward = np.ones([1, commands_compose_each]) * yaw_forward
                    data3_yaw_backward = np.ones([1, commands_compose_each]) * yaw_backward
                    data3_yaw_leftward = np.ones([1, commands_compose_each]) * yaw_leftward
                    data3_yaw_rightward = np.ones([1, commands_compose_each]) * yaw_rightward
                    data3_to_feed = np.concatenate([data3_yaw_forward, data3_yaw_backward, data3_yaw_leftward, data3_yaw_rightward], axis=1)

                    results = sess.run(result, feed_dict={rgb_data: rgb_image, line_data_2: data3_to_feed})
                    # cv2.imshow("rgb2", rgb_image[0,:,:,:])
                    # cv2.waitKey(5)

                    images = sess.run(image_to_show, feed_dict={rgb_data: rgb_image, line_data_2: data3_to_feed})
                    merged = merge_images(images, 2)

                    cv2.namedWindow("merged_rgb", cv2.WINDOW_NORMAL)
                    cv2.imshow("merged_rgb", merged)
                    cv2.waitKey()

                    move_cmd.linear.x = 0.2#results[0, 0] * 0.6  # 1.0

                    move_cmd.angular.z = (2 * results[0, 1] - 1) * 1.0  # 0.88

                cmd_pub.publish(move_cmd)

                # if the .py is running
                status_flag.data += 0.1
                if status_flag.data > 1000.0:
                    status_flag.data = 1000.0
                status_pub.publish(status_flag)

                print ("F-B-L-R:", yaw_forward, yaw_backward, yaw_leftward, yaw_rightward)
                # print yaw_forward, yaw_backward, yaw_leftward, yaw_rightward
                # print ("forward/backward/yaw_leftward/yaw_rightward: ", data3_to_feed)
                print ("linear_vel, angular-vel: ", move_cmd.linear.x, move_cmd.angular.z)

                new_msg_received = False
            rate.sleep()

