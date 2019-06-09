import tensorflow as tf
import math
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import cv2
import time
import csv
import random
from scipy.ndimage.filters import maximum_filter

fileobj = open('test_depth.csv', 'wb')
file_writer = csv.writer(fileobj)

commands_compose_each = 1  # Should be "input3_dim": 8  / 4

# model_path = "/home/ubuntu/chg_workspace/depth/model/cnn_nornn/01_noi_rotate/model/simulation_cnn_rnn_iter3.ckpt"
model_path = "/home/ubuntu/chg_workspace/depth/model/cnn_nornn/02_noi_rotate/model/simulation_cnn_rnn_iter3.ckpt"



''' Parameters for input vectors'''
input_paras = {
    "input1_dim_x": 256,
    "input1_dim_y": 192,
    "input1_dim_channel": 1,
    "input2_dim": 4  # commands
}

commands_compose_each = 1  # Should be "input3_dim": 4  / 4

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

rgb_image = np.zeros([1, img_height, img_wid, img_channel])  # bgr in opencv form
bridge = CvBridge()


'''threshold for deciding if adding noise to the images'''
thres_gaussian = 0.5
thres_pepper = 0.2


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


class Networkerror(RuntimeError):
    """
    Error print
    """
    def __init__(self, arg):
        self.args = arg


def callBackDepth(img):
    try:
        cv_image = bridge.imgmsg_to_cv2(img)
    except CvBridgeError as e:
        print(e)

    global rgb_image, new_msg_received

    image_mm = cv2.resize(cv_image, (img_wid, img_height), interpolation=cv2.INTER_AREA)

    rgb_image_list = image_mm * 0.001  # mm -> m
    rgb_image_list[rgb_image_list > 4.5] = 0.0
    rgb_image_list[np.isnan(rgb_image_list)] = 0.0

    image_mm = rgb_image_list * 1000

    rgb_image_uint = np.trunc(image_mm * 56).astype(np.uint8)
    # rgb_image_uint = np.array(rgb_image_uint).reshape(1, img_height, img_wid, img_channel)

    rgb_image = np.array(rgb_image_uint).reshape(1, img_height, img_wid, img_channel)

    # gaussian random noise generation: add to the origin image with the unit of mm
    # sigma_l_mm = np.ones(image_mm.shape) * 3.1

    # image_with_contour_noise = rgb_image_uint[0, :, :, :]
    #
    # cv2.imshow("img_origin_unit8", rgb_image_uint[0, :, :, :])
    # cv2.waitKey(5)
    #
    # if np.random.random_sample() < thres_gaussian:
    #     image_with_contour_noise = ContourGaussianNoise(image_mm, rgb_image_uint, contour_width=5)
    #
    # if np.random.random_sample() < thres_pepper:
    #     pepper_percentage = np.random.random_sample() / 2
    #     SaltAndPepper(image_with_contour_noise, pepper_percentage)
    #
    # cv2.imshow("depth contour", image_with_contour_noise)
    # cv2.waitKey(5)

    # line_to_save_as_csv = np.zeros([img_height * img_wid, 1])
    # mat_to_save = np.zeros([img_height * img_wid, 1])
    # for i in range(img_height):
    #     for j in range(img_wid):
    #         line_to_save_as_csv[i * img_wid + j, 0] = rgb_image[0, i, j, 0]
    # mat_to_save = np.concatenate((mat_to_save, line_to_save_as_csv), axis=0)
    # with open('depth_short12.csv', 'wb') as depth_csv:
    #     np.savetxt(depth_csv, mat_to_save, delimiter=',')
    # file_writer.writerow(line_to_save_as_csv, delimiter=',')

    # cv2.imshow("depth", rgb_image)
    # cv2.waitKey(5)
    new_msg_received = True


def ContourGaussianNoise(src, rgb_image_uint8, contour_width):
    image_mm = src
    # parameters for the gaussian noise at the contour
    theta_mean = 3.1415926 / 6
    sigma_z_mm = 1.5 - 0.5 * image_mm + 0.3 * image_mm * image_mm \
                 + 0.1 * np.power(image_mm, 1.5) * theta_mean * theta_mean \
                 / (3.1415 - theta_mean) * (3.1415 - theta_mean)

    randoms_normal = np.random.randn(image_mm.shape[0], image_mm.shape[1])

    noise_z_mm = randoms_normal * sigma_z_mm + image_mm
    noise_z_m_uint8 = np.trunc(noise_z_mm * 0.001 * 40).astype(np.uint8)

    # noise_z_mm = np.exp(-randoms / (2 * sigma_z_mm * sigma_z_mm))

    # reshape the image in the unit mm
    # image_mm = np.array(image_mm).reshape(1, img_height, img_wid, img_channel)

    # find contours
    contours = cv2.Canny(rgb_image_uint8[0, :, :, :], 50, 500)

    _, contours_bin, hierarchy = cv2.findContours(contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = cv2.findContours(rgb_image_uint[0, :, :, :])

    cv2.drawContours(contours, contours_bin, -1, 255, 8)

    # the contours with noise
    contour_wider_uint8 = cv2.bitwise_and(noise_z_m_uint8, contours)

    # contour_wider_noise = contour_wider_uint8 * noise_z_mm
    # contour_wider_noise = np.trunc(contour_wider_noise * 0.001 * 40).astype(np.uint8)

    contour_not = cv2.bitwise_not(contour_wider_uint8)

    image_without_contour = cv2.bitwise_and(contour_not, rgb_image_uint8[0, :, :, 0])

    image_with_contour_noise = image_without_contour + contour_wider_uint8

    return image_with_contour_noise


# def SaltAndPepper(src, percetage):
#     SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
#     for i in range(SP_NoiseNum):
#         randX = random.randint(0, src.shape[0]-1)
#         randY = random.randint(0, src.shape[1]-1)
#         if np.random.random_integers(0, 1) == 0:
#             src[randX, randY] = 0
#         else:
#             src[randX, randY] = 255
#     return src


def SaltAndPepper(src, percetage):
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = np.random.normal(loc=0.0, scale=img_height/24, size=None)
        randY = np.random.normal(loc=0.0, scale=img_wid/24, size=None)

        randX = np.maximum(randX, - img_height / 2)
        randX = np.minimum(randX, img_height / 2)

        randY = np.maximum(randY, - img_wid / 2)
        randY = np.minimum(randY, img_wid / 2)

        margin_y = np.random.randint(0, int(img_wid))
        margin_x = np.random.randint(0, int(img_height))

        if randX <= 0:
            randX = img_height + randX
        if randY <= 0:
            randY = img_wid + randY
        randX = int(randX)
        randY = int(randY)

        if np.random.random_integers(0, 1) == 0:
            src[randX, margin_y] = 0

        # if np.random.random_integers(0, 1) == 0:
        #     src[randX, margin_y] = 0

        if np.random.random_integers(0, 1) == 0:
            src[margin_x, randY] = 0

        # if np.random.random_integers(0, 1) == 0:
        #     src[margin_x, randY] = 0

    return src


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
    rospy.Subscriber("/camera/depth/image_raw", Image, callBackDepth)
    rospy.Subscriber("/radar/delt_yaw", Float64, callBackDeltYaw)
    rospy.Subscriber("/radar/current_yaw", Float64, callBackCurrentYaw)
    rospy.Subscriber("/odom", Odometry, callBackOdom)
    cmd_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)
    move_cmd = Twist()
    status_pub = rospy.Publisher("/robot/status", Float64, queue_size=1)
    status_flag = Float64()
    status_flag.data = 0.0

    ''' Graph building '''
    rgb_data = tf.placeholder("float", name="rgb_data", shape=[None, input_dimension_y, input_dimension_x,
                                                               input_channel])
    line_data_2 = tf.placeholder("float", name="line_data", shape=[None, input_paras["input2_dim"]])  # commands

    # 3D CNN
    encode_vector = encoder(rgb_data)
    print "encoder built"
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
    print "concat complete"

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

        print "parameters restored!"
        global new_msg_received

        while not rospy.is_shutdown():
            if new_msg_received:
                data3_yaw_forward = np.ones([1, commands_compose_each]) * yaw_forward
                data3_yaw_backward = np.ones([1, commands_compose_each]) * yaw_backward
                data3_yaw_leftward = np.ones([1, commands_compose_each]) * yaw_leftward
                data3_yaw_rightward = np.ones([1, commands_compose_each]) * yaw_rightward
                data3_to_feed = np.concatenate([data3_yaw_forward, data3_yaw_backward, data3_yaw_leftward, data3_yaw_rightward], axis=1)

                results = sess.run(result, feed_dict={rgb_data: rgb_image, line_data_2: data3_to_feed})
                # cv2.imshow("rgb2", rgb_image[0,:,:,:])
                # cv2.waitKey(5)

                move_cmd.linear.x = results[0, 0] * 0.8  # 1.0

                move_cmd.angular.z = (2 * results[0, 1] - 1) * 1.0  # 0.88

                cmd_pub.publish(move_cmd)

                # if the .py is running
                status_flag.data += 0.1
                if status_flag.data > 1000.0:
                    status_flag.data = 1000.0
                status_pub.publish(status_flag)

                # print yaw_forward, yaw_backward, yaw_leftward, yaw_rightward
                print data3_to_feed
                print results

                new_msg_received = False
            rate.sleep()
