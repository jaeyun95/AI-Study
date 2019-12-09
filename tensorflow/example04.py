#(4) example04
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
weight1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
weight2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

bias1 = tf.Variable(tf.zeros([10]))
bias2 = tf.Variable(tf.zeros([3]))

#activation function
layer1 = tf.add(tf.matmul(X, weight1), bias1)
layer2 = tf.nn.relu(layer1)

model = tf.add(tf.matmul(layer1, weight2), bias2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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