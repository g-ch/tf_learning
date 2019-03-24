import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

keep_prob = tf.placeholder("float")

k_dia = {"a": 32}

def conv_relu(x, kernel_shape, bias_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_con", bias_shape, initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(x, weights, strides=strides, padding="SAME")
    return tf.nn.relu(conv + biases)


def deconv(x, kernel_shape, output_shape, strides):
    weights = tf.get_variable("weights_con", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.conv2d_transpose(x, filter=weights, output_shape=output_shape, strides=strides)


def max_pool(x, kernel_shape, strides):
    return tf.nn.max_pool(x, ksize=kernel_shape, strides=strides, padding='SAME')


def relu(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights_relu", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_relu", [neurals_num], initializer=tf.constant_initializer(0.1))
    return tf.nn.relu(tf.matmul(x, weights) + biases)


def softmax(x, x_diamension, neurals_num):
    weights = tf.get_variable("weights_soft", [x_diamension, neurals_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("bias_soft", [neurals_num], initializer=tf.constant_initializer(0.1))
    return tf.nn.softmax(tf.matmul(x, weights) + biases)


def cnn_encoder(x):
    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            relu1 = conv_relu(x, kernel_shape=[3, 3, 1, k_dia["a"]], bias_shape=[32], strides=[1, 1, 1, 1])
            pool1 = max_pool(relu1, kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        with tf.variable_scope("conv2"):
            relu2 = conv_relu(pool1, kernel_shape=[3, 3, 32, 64], bias_shape=[64], strides=[1, 1, 1, 1])
            return max_pool(relu2, kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1])


def cnn_decoder(x):
    with tf.variable_scope("decoder"):
        with tf.variable_scope("conv1"):
            relu1 = conv_relu(x, kernel_shape=[3, 3, 64, 64], bias_shape=[64], strides=[1, 1, 1, 1])
        with tf.variable_scope("deconv1"):
            deconv1 = deconv(relu1, kernel_shape=[2, 2, 64, 64], output_shape=[100, 14, 14, 64], strides=[1, 2, 2, 1])
        with tf.variable_scope("conv2"):
            relu2 = conv_relu(deconv1, kernel_shape=[3, 3, 64, 32], bias_shape=[32], strides=[1, 1, 1, 1])
        with tf.variable_scope("deconv2"):
            deconv2 = deconv(relu2, kernel_shape=[2, 2, 32, 32], output_shape=[100, 28, 28, 32], strides=[1, 2, 2, 1])
        with tf.variable_scope("recovery"):
            return conv_relu(deconv2, kernel_shape=[3, 3, 32, 1], bias_shape=[1], strides=[1, 1, 1, 1])


def cnn_autoencoder():
    ''' Autoencoder Test '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    learning_rate = 1e-4
    batch_size = 100

    x_ = tf.placeholder("float", shape=[None, 784])
    input_image = tf.reshape(x_, [-1, 28, 28, 1])

    encode_output = cnn_encoder(input_image)

    cnn_flat = tf.reshape(encode_output, [-1, 7 * 7 * 64])

    with tf.variable_scope("fully_encoder"):
        fully_encoder = relu(cnn_flat, 7 * 7 * 64, 4)
    with tf.variable_scope("fully_decoder"):
        fully_decoder = relu(fully_encoder, 4, 7 * 7 * 64)

    cnn_recover = tf.reshape(fully_decoder, [-1, 7, 7, 64])

    decode_image = cnn_decoder(cnn_recover)

    loss = tf.reduce_mean(tf.square(input_image - decode_image))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print "Start training"
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / batch_size)

        for epoch in range(10):
            for i in range(total_batch):
                batch = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x_: batch[0]})  # training

            print('loss=%s' % sess.run(loss, feed_dict={x_: batch[0]}))
            saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/mnist.ckpt')

        # Test
        # test_total_batch = int(mnist.test.num_examples / batch_size)
        test_batch = mnist.test.next_batch(batch_size)

        recon = sess.run(decode_image, feed_dict={x_: test_batch[0]})

        # draw
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(test_batch[0][i], (28, 28)))
            a[1][i].imshow(np.reshape(recon[i], (28, 28)))
        plt.show()
        print('test loss=%s' % sess.run(loss, feed_dict={x_: test_batch[0]}))  # Test


def fully_connected(x, input_dim, output_dim):
    with tf.variable_scope("fully"):
        with tf.variable_scope("relu1"):
            sf_1 = relu(x, x_diamension=input_dim, neurals_num=1024)
        with tf.variable_scope("dropout"):
            h_fc1_drop = tf.nn.dropout(sf_1, keep_prob)
        with tf.variable_scope("softmax"):
            return softmax(h_fc1_drop, x_diamension=1024, neurals_num=output_dim)
        # with tf.variable_scope("softmax"):
        #     return softmax(x, x_diamension=input_dim, neurals_num=output_dim)


def classify_test():
    ''' Classfication Test '''

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    learning_rate = 1e-4
    batch_size = 100

    x_ = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    input_image = tf.reshape(x_, [-1, 28, 28, 1])
    cnn_relu = cnn_encoder(input_image)
    cnn_relu_flat = tf.reshape(cnn_relu, [-1, 7 * 7 * 64])

    output = fully_connected(cnn_relu_flat, input_dim=7 * 7 * 64, output_dim=10)

    # output = fully_connected(x_, input_dim=784, output_dim=10)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(output))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "Start training"
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / batch_size)

        for epoch in range(10):
            for i in range(total_batch):
                batch = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x_: batch[0], y_: batch[1], keep_prob: 0.5})  # training

            print('acc=%s' % sess.run(accuracy, feed_dict={x_: batch[0], y_: batch[1], keep_prob: 1.0}))
            saver.save(sess, '/home/ubuntu/chg_workspace/3dcnn/model/mnist.ckpt')

        print('test acc=%s' % sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels,
                                                            keep_prob: 1.0}))  # Test


if __name__ == '__main__':
    # classify_test()
    cnn_autoencoder()