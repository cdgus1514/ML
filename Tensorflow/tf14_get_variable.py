import tensorflow as tf
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)

# Data load
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images)
print(mnist.test.labels)
print(mnist.train.images.shape) # (55000, 784)
print(mnist.test.labels.shape)  # (10000, 10)



# Graph
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

## 변수구성하면서 초기화 진행
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.leaky_relu(tf.matmul(X,W1)+b1)
# L1 = tf.nn.dropout(L1,0.5)

# W2 = tf.Variable(tf.random_normal([256,512]))
W2 = tf.get_variable("W2", shape=[256,512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.leaky_relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2,0.5)

# W3 = tf.Variable(tf.random_normal([512,10]))
W3 = tf.get_variable("W3", shape=[512,10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.nn.softmax(tf.matmul(L2,W3)+b3)
hypothesis = tf.matmul(L2,W3)+b3



# tf.constant_initializer()
# tf.zeros_initializer()
# tf.random_uniform_initializer()
# tf.random_uniform_initializer()


# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
# train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
train = tf.train.AdamOptimizer(0.01).minimize(cost)

## Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

## parameter
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)



# launch graph
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
    # plt.show()