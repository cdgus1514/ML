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


L1 = tf.layers.dense(X, 256, activation=tf.nn.relu)
L1 = tf.layers.dropout(L1, 0.2)

L2 = tf.layers.dense(L1, 512, activation=tf.nn.relu)
L2 = tf.layers.dropout(L2, 0.2)

L3 = tf.layers.dense(L2, 1024, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.2)

logits = tf.layers.dense(L3, 1, activation=None)



cost = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.train.AdamOptimizer(0.001).minimize(cost)

predicted = tf.cast(logits > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))



# model launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_train, Y:y_train})

        if step % 10 == 0:
            print(step, cost_val)

    # Acuuracy
    h, c, a = sess.run([logits, predicted, accuracy], feed_dict={X:x_test, Y:y_test})
    print("Hypothesis :\n", h, "\nCorrect (Y) :\n", c, "\nAccuracy :", a)
