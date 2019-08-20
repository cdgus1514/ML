import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# data load
xy = np.loadtxt("C:/Study/ML/Data/data-03-diabetes.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape) # (759,8)
print(y_data.shape) # (759,1)


X = tf.placeholder(tf.float32, shape=[None,8])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


# model(graph)
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)


# model compile
## loss="binary_crossentropy"
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    # Acuuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("Hypothesis :\n", h, "\nCorrect (Y) :\n", c, "\nAccuracy :", a)


'''
Accuracy : 0.7628459
'''