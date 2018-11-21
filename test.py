import tensorflow as tf

W = tf.Variable(1.0, name="W")
double = tf.multiply(2.0, W)

saver = tf.train.Saver({'weights':W})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4):
        sess.run(tf.assign_add(W, 1.0))
        saver.save(sess, '/home/ubuntu/chg_workspace/data/test.ckpt')
        print('W=%s, double=%s' % (sess.run(W), sess.run(double)))
