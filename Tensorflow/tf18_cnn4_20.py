import tensorflow as tf
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)

# Data load
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

lr = 0.001
epochs = 15
batch_size = 100



# Graph
X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, [-1,28,28,1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

# W1 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))
# L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME")
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

## layer1
L1 = tf.layers.conv2d(X_img, 32, [3,3], activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2,2], [2,2])
L1 = tf.layers.dropout(L1, 0.2)

# layer2
L2 = tf.layers.conv2d(L1, 64, [3,3], activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2,2], [2,2])


# # from tensorflow.keras import layers
# L3 = tf.layers.conv2d(L2, 32, [3,3], activation=tf.nn.relu)
# L3 = tf.layers.max_pooling2d(L3, [2,2], [2,2])
# L3 = tf.layers.dropout(L3, 0.7)

L4 = tf.contrib.layers.flatten(L2)
L4 = tf.layers.dense(L4, 256, activation=tf.nn.relu)
L4 = tf.layers.dropout(L4, 0.5)

L2_flat = tf.layers.dense(L4, 10, activation=None)
# L2_flat = tf.reshape(L2_flat, [-1,7*7*64])



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L2_flat, labels=Y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


print("Learning started.")
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)    # (55000/100)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # print(batch_xs.shape) # (100,784)
        # print(batch_ys.shape) # (100,10)

        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print("Epochs >> ", "%04d" % (epoch+1), "cost >> ", "{:.9f}".format(avg_cost))

print("Finish")


# accuracy
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))
correct_prediction = tf.equal(tf.argmax(L2_flat, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy >>", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))


# predict
r = random.randint(0, mnist.test.num_examples -1)
print("Label >>", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction >>", sess.run(tf.argmax(mnist.test.images[r:r+1])))
