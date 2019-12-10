#(10) example10
#import tensorflow and numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#this is for load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

#load mnist dataset
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#define option
total_epoch = 10
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128

#input size is [28X28]
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])


#make generater parameter
G_weight1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_bias1 = tf.Variable(tf.zeros([n_hidden]))
G_weight2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_bias2 = tf.Variable(tf.zeros([n_input]))

#make discriminator parameter
D_weight1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_bias1 = tf.Variable(tf.zeros([n_hidden]))
D_weight2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_bias2 = tf.Variable(tf.zeros([1]))

#make is generator network
def generator(noise):
    hidden = tf.nn.relu(tf.matmul(noise, G_weight1) + G_bias1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_weight2) + G_bias2)
    return output


#make discriminator network
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_weight1) + D_bias1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_weight2) + D_bias2)
    return output

#make model
Generator = generator(Z)
Discriminator_generation = discriminator(Generator)
Discriminator_real = discriminator(X)

loss_Discriminator = tf.reduce_mean(tf.log(Discriminator_real) + tf.log(1 - Discriminator_generation))
loss_Generator = tf.reduce_mean(tf.log(Discriminator_generation))

Discriminator_var_list = [D_weight1, D_bias1, D_weight2, D_bias2]
Generator_var_list = [G_weight1, G_bias1, G_weight2, G_bias2]

train_Discriminator = tf.train.AdamOptimizer(learning_rate).minimize(-loss_Discriminator,var_list=Discriminator_var_list)
train_Generator = tf.train.AdamOptimizer(learning_rate).minimize(-loss_Generator,var_list=Generator_var_list)
       

#training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#setting batch size
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = np.random.normal(size=(batch_size, n_noise))
        _, loss_val_Discriminator = sess.run([train_Discriminator, loss_Discriminator], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_Generator = sess.run([train_Generator, loss_Generator], feed_dict={Z: noise})
    print('Epoch:', '%04d' % epoch,'D loss: {:.4}'.format(loss_val_Discriminator),'G loss: {:.4}'.format(loss_val_Generator))
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = np.random.normal(size=(batch_size, n_noise))
        samples = sess.run(Generator, feed_dict={Z: noise})
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        plt.savefig('{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)