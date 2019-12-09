#(1) example01
#import tensorflow
import tensorflow as tf

#make  constant
a = tf.constant(30)
b = tf.constant([1,2])
c = tf.constant([[1,2],[3,4]])
d = tf.constant([[[1,2],[3,4]]])
e = tf.constant('tensorflow test!')

#check your example
print(a)
print(b)
print(c)
print(d)
print(e)

#make graph
param1 = tf.constant(10)
param2 = tf.constant(20)
param3 = tf.add(param1,param2)
sess = tf.Session()
result = sess.run(param3)

#check your example
print(param3)
print(result)

#make placeholder
param4 = tf.placeholder(tf.float32,[None,3])

#check your example
print(param4)

#make variable
f = tf.Variable(tf.random_normal([2,2]))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#check your example
print(sess.run(f))

#make variable and constant
X = tf.placeholder(tf.float32, [None, 3])
x_data = [[1, 2, 3], [4, 5, 6]]
sess = tf.Session()

weight = tf.Variable(tf.random_normal([3,2]))
bias = tf.Variable(tf.random_normal([2,1]))

fx = tf.matmul(X,weight)+bias

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ans = sess.run(fx,feed_dict={X:x_data})

#check your example
print(ans)



