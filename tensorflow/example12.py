#(12) example12
#import tensorflow and numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#this is for load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

#load mnist dataset
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#define option
learning_rate = 0.001
total_epoch = 20
batch_size = 128

n_input = 28 #this is cell input size
n_step = 28 #define step number
n_hidden = 128
n_class = 10

#input size is [28,28] 
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

#define cell 
#cell function : BasicRNNCell,BasicLSTMCell,GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

##make RNN model
'''
if you don't use pre-defined function

states = tf.zeros(batch_size)
for i in range(n_step):
    outputs, states = cell(X[[:, i]], states)
'''
#if you use pre-defined function
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#Y's shape : [batch_size, n_class]
#output's shape : [batch_size, n_step, n_hidden]
#we will change shape like below
#[batch_size, n_step, n_hidden] -> [n_step, batch_size, n_hidden] -> [batch_size, n_hidden]

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#setting batch size
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #change shape
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#check result
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels