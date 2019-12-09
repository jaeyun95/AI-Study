#(2) example02
#import tensorflow
import tensorflow as tf

#optimizer
x_data = [1,2,3]
y_data = [1,2,3]

weight = tf.Variable(tf.random_uniform([1],-1.,1.))
bias = tf.Variable(tf.random_uniform([1],-1.,1.))

X = tf.placeholder(tf.float32,name="X")
Y = tf.placeholder(tf.float32,name="Y")

fx = weight * X + bias

cost = tf.reduce_mean(tf.square(fx-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        #check your example
        print(step, cost_val, sess.run(weight), sess.run(bias))

    #check your example
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(fx, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(fx, feed_dict={X: 2.5}))