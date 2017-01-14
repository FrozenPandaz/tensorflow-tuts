# Import the image data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Begin Model
x = tf.placeholder(tf.float32, [None, 784])

# Setup Weights
W = tf.Variable(tf.zeros([784, 10]))
# Wx will give a 1x10

# Setup biases
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# End Model

# Actual Results
y_ = tf.placeholder(tf.float32, [None, 10])

# Mean of the sum of y*log(y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Optimize for the lowest cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Define initial variables
init = tf.global_variables_initializer()

# Start a session
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
