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

L1 = tf.layers.dense(X, 256, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 512, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 256, activation=tf.nn.relu)
L4 = tf.layers.dense(L3, 512, activation=tf.nn.relu)

logits = tf.layers.dense(L4, 1, activation=None)

cost = tf.reduce_mean(tf.square(y_train - logits))
train = tf.train.RMSPropOptimizer(0.001).minimize(cost)

prediction = tf.argmax(logits, 1)
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
    h, c, a = sess.run([logits, is_correct, accuracy], feed_dict={X:x_test, Y:y_test})
    # print("Hypothesis :\n", h, "\nCorrect (Y) :\n", c, "\nAccuracy :", a)
    print("\nAccuracy :", accuracy.eval(session=sess, feed_dict={X:x_test, Y:y_test}))


    from sklearn.metrics import mean_squared_error
    def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))

    print("RMSE : ", RMSE(y_test, sess.run(is_correct, feed_dict={X:x_test, Y:y_test})))



    from sklearn.metrics import r2_score
    r2_y_predict = r2_score(y_test, sess.run(is_correct, feed_dict={X:x_test, Y:y_test}))
    print("\nR2 : ", r2_y_predict)