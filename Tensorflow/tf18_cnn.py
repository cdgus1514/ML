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

lr = 0.001
epochs = 15
batch_size = 100



# Graph
X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, [-1,28,28,1]) # b/w
Y = tf.placeholder(tf.float32, shape=[None, 10])

## layer1
W1 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))     # kernel_size = 3,3 // 흑백 = 1 // output
# print(W1) # <tf.Variable 'Variable:0' shape=(3, 3, 1, 32) dtype=float32_ref>
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME") # 1,1 >> 1칸씩 이동 + padding >> (28,28,32)
# print(L1) # Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") # (2,2) cut >> 2칸씩 이동 >> (14,14,32)
# print(L1) # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)


# layer2
W2 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01))    # 32 == W1의 output
# print(W2) # <tf.Variable 'Variable_1:0' shape=(3, 3, 32, 64) dtype=float32_ref>
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding="SAME")
# print(L2) # Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# print(L2) # ensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

L2_flat = tf.reshape(L2, [-1,7*7*64])