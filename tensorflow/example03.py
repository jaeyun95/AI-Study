#(3) example03
#import tensorflow and numpy
import tensorflow as tf
import numpy as np

#[feather, wing]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

#[etc, mammal, bird]
#one-hot encoding(label)
y_data = np.array([
    [1, 0, 0],  #etc
    [0, 1, 0],  #mammal
    [0, 0, 1],  #bird
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

#make simple model
#make placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#input size is 2, output size is 3
weight = tf.Variable(tf.random_uniform([2, 3], -1., 1.))

bias = tf.Variable(tf.zeros([3]))

layer = tf.add(tf.matmul(X, weight), bias)
#activation function
layer = tf.nn.relu(layer)

model = tf.nn.softmax(layer)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
prediction = tf.argmax(model, 1)
ground_truth = tf.argmax(Y, 1)
print('Prediction:', sess.run(prediction, feed_dict={X: x_data}))
print('Ground Truth:', sess.run(ground_truth, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, ground_truth)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))