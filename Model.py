from CifarDataManager import CifarDataManager
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from Helper import Helper as hp
import numpy as np

cifar = CifarDataManager()
hp = hp()

STEPS = 1000
BTACH_SIZE = 500

tf.reset_default_graph()


# Model IP/OP dif.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="output")
keep_prob = tf.placeholder(tf.float32)

#1st conv_layer
conv1 = hp.conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = hp.max_pool_2x2(conv1)

conv2 = hp.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = hp.max_pool_2x2(conv2)

conv3 = hp.conv_layer(conv2_pool, shape=[5, 5, 64, 128])
conv3_pool = hp.max_pool_2x2(conv3)

conv3_flat = tf.reshape(conv3_pool, [-1, 4*4*128])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full_1 =tf.nn.relu(hp.full_layer(conv3_drop, 512))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv =hp.full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)



def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x:X[i], y_:Y[i], keep_prob:1.0}) for i in range(10)])
    print("Accuracy :{:.4}%".format(acc*100))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        if i % 10 == 0:
            batch = cifar.train.next_batch(BTACH_SIZE)
            summary = sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    test(sess)


