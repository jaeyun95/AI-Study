#(7) example07
#import tensorflow and numpy
import tensorflow as tf
#this is for load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

#load mnist dataset
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# input size is [28X28]
X = tf.placeholder(tf.float32, [None, 784])
# 0~10 classification
Y = tf.placeholder(tf.float32, [None, 10])


weight1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
layer1 = tf.nn.relu(tf.matmul(X, weight1))

weight2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
layer2 = tf.nn.relu(tf.matmul(layer1, weight2))

weight3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(layer2, weight3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#setting batch size
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))


is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels}))