import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images)
print(mnist.test.labels)
print(mnist.train.images.shape) # (55000, 784)
print(mnist.test.labels.shape)  # (10000, 10)

#################################################################
#       코딩 X,Y,W,b,hypothesis, cost, train                    #
#################################################################

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal([784,10], name="weight"))
b = tf.Variable(tf.random_normal([10], name="bias"))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(0.002).minimize(cost)
# train = tf.train.AdamOptimizer(1e-50).minimize(cost)



# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameter
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epochs: {:04d}, Cost: {:.9f}".format(epochs+1, avg_cost))
        
    print("Learning finished")

    
    # Test the model using test sets
    print("Accuracy :", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples-1)
    print("Label :", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction :", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}),)

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation="nearest")
    plt.show()