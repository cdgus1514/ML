import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# Data Load
xy = np.load("cancer_train.npz")
x_train = xy["x_train"]
y_train = xy["y_train"]
xy = np.load("cancer_test.npz")
x_test = xy["x_test"]
y_test = xy["y_test"]

y_train = np.reshape(y_train, (455,1))
y_test = np.reshape(y_test, (114,1))

print(x_train.shape, y_train.shape) # (455, 30) (455, 1)
print(x_test.shape, y_test.shape) # (114, 30) (114, 1)



# Model
X = tf.placeholder(tf.float32, shape=[None,30])
Y = tf.placeholder(tf.float32, shape=[None,1])

## 변수구성하면서 초기화 진행
W1 = tf.get_variable("W1", shape=[30, 32], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([32]))
L1 = tf.nn.leaky_relu(tf.matmul(X,W1)+b1)
# L1 = tf.nn.dropout(L1,0.5)

W2 = tf.get_variable("W2", shape=[32,64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([64]))
L2 = tf.nn.leaky_relu(tf.matmul(L1,W2)+b2)
# L2 = tf.nn.dropout(L2,0.5)

W3 = tf.get_variable("W3", shape=[64,1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.nn.softmax(tf.matmul(L2,W3)+b3)
# hypothesis = tf.matmul(L2,W3)+b3
hypothesis = tf.sigmoid(tf.matmul(L2,W3)+b3)

cost = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y))
train = tf.train.AdamOptimizer(0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))



# model launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_train, Y:y_train})

        if step % 10 == 0:
            print(step, cost_val)

    # Acuuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_test, Y:y_test})
    print("Hypothesis :\n", h, "\nCorrect (Y) :\n", c, "\nAccuracy :", a)
    # print("Accuracy :", accuracy.eval(session=sess, feed_dict={X:x_test, Y:y_test}))


'''
Accuracy : 0.92105263
'''