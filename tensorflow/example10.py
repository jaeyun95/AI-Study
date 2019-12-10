#(10) example10
#import tensorflow and numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#this is for load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

#load mnist dataset
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#input size is [28X28]
X = tf.placeholder(tf.float32, [None, 28*28])

#make encoder
encoder_weight = tf.Variable(tf.random_normal([28*28, 256]))
encoder_bias = tf.Variable(tf.random_normal([256]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_weight), encoder_bias))

#make decoder
decoder_weight = tf.Variable(tf.random_normal([256, 28*28]))
decoder_bias = tf.Variable(tf.random_normal([28*28]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, decoder_weight), decoder_bias))

cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#setting batch size
batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(20):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

sample_size = 10

samples = sess.run(decoder,
                   feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()
