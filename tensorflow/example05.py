#(5) example05
#import tensorflow and numpy
import tensorflow as tf
import numpy as np

#data loading
data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

#make simple model
#make placeholder
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#input size is 2, output size is 3
weight1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
weight2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
weight3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))

#activation function
layer1 = tf.matmul(X, weight1)
layer2 = tf.nn.relu(tf.matmul(layer1, weight2))

model = tf.matmul(layer2, weight3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)


sess = tf.Session()

#save model
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exixts(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

#optimizer
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
saver.save(sess, './model/dnn.ckpt',global_step=global_step)


prediction = tf.argmax(model, 1)
ground_truth = tf.argmax(Y, 1)
print('Prediction:', sess.run(prediction, feed_dict={X: x_data}))
print('Ground Truth:', sess.run(ground_truth, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, ground_truth)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))