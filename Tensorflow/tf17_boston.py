import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# Data Load
xy = np.load("boston_housing_train.npz")
x_train = xy["x_train"]
y_train = xy["y_train"]
xy = np.load("boston_housing_test.npz")
x_test = xy["x_test"]
y_test = xy["y_test"]

y_train = np.reshape(y_train, (404, 1))
y_test = np.reshape(y_test, (102, 1))

print(x_train.shape, y_train.shape) # (404, 13) (404, 1)
print(x_test.shape, y_test.shape)   # (102, 13) (102, 1)



# Graph
X = tf.placeholder(tf.float32, shape=[None,13])
Y = tf.placeholder(tf.float32, shape=[None,1])

## 변수구성하면서 초기화 진행
W1 = tf.get_variable("W1", shape=[13, 32], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([32]))
L1 = tf.nn.leaky_relu(tf.matmul(X,W1)+b1)
# L1 = tf.nn.dropout(L1,0.5)

W2 = tf.get_variable("W2", shape=[32,128], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]))
L2 = tf.nn.leaky_relu(tf.matmul(L1,W2)+b2)
# L2 = tf.nn.dropout(L2,0.5)

W3 = tf.get_variable("W3", shape=[128,1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.nn.softmax(tf.matmul(L2,W3)+b3)
# hypothesis = tf.matmul(L2,W3)+b3
hypothesis = tf.matmul(L2,W3)+b3

# cost = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y))
cost = tf.reduce_mean(tf.square(y_train - hypothesis))
# train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# train = tf.train.AdamOptimizer(0.001).minimize(cost)
train = tf.train.RMSPropOptimizer(0.001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))



# model launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_train, Y:y_train})

        if step % 10 == 0:
            print(step, cost_val)

    # Acuuracy
    h, c, a = sess.run([hypothesis, is_correct, accuracy], feed_dict={X:x_test, Y:y_test})
    # print("Hypothesis :\n", h, "\nCorrect (Y) :\n", c, "\nAccuracy :", a)
    print("\nAccuracy :", accuracy.eval(session=sess, feed_dict={X:x_test, Y:y_test}))


    from sklearn.metrics import mean_squared_error
    def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))

    print("RMSE : ", RMSE(y_test, sess.run(is_correct, feed_dict={X:x_test, Y:y_test})))



    from sklearn.metrics import r2_score
    r2_y_predict = r2_score(y_test, sess.run(is_correct, feed_dict={X:x_test, Y:y_test}))
    print("\nR2 : ", r2_y_predict)