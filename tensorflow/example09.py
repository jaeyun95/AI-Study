#(9) example09
#import tensorflow and numpy
import tensorflow as tf
import numpy as np
#this is for load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

#load mnist dataset
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#input size is [28X28]
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

#make noise
noise = np.random.random((28, 28, 1))
noise = np.repeat(noise,100)
noise = noise.reshape(-1, 28, 28, 1)

#0~10 classification
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

weight1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
layer1 = tf.nn.conv2d(X, weight1, strides=[1, 1, 1, 1], padding='SAME')
layer1 = tf.nn.relu(layer1)
layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

weight2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
layer2 = tf.nn.conv2d(layer1, weight2, strides=[1, 1, 1, 1], padding='SAME')
layer2 = tf.nn.relu(layer2)
layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


weight3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
layer3 = tf.reshape(layer2, [-1, 7 * 7 * 64])
layer3 = tf.matmul(layer3, weight3)
layer3 = tf.nn.relu(layer3)
layer3 = tf.nn.dropout(layer3, keep_prob)

weight4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(layer3, weight4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#setting batch size
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(5):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))


is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob: 1}))