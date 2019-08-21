import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# Dataset
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)



# model
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2,11]), name="weight1")
b1 = tf.Variable(tf.random_normal([11]), name="bias1")
L1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W1 = tf.Variable(tf.random_normal([2,100]), name="weight2")
b1 = tf.Variable(tf.random_normal([100]), name="bias2")
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,200]), name="weight3")
# b1 = tf.Variable(tf.random_normal([200]), name="bias3")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,300]), name="weight4")
# b1 = tf.Variable(tf.random_normal([300]), name="bias4")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,400]), name="weight5")
# b1 = tf.Variable(tf.random_normal([400]), name="bias5")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,500]), name="weight6")
# b1 = tf.Variable(tf.random_normal([500]), name="bias6")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,600]), name="weight7")
# b1 = tf.Variable(tf.random_normal([600]), name="bias7")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,700]), name="weight8")
# b1 = tf.Variable(tf.random_normal([700]), name="bias8")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# W1 = tf.Variable(tf.random_normal([2,800]), name="weight9")
# b1 = tf.Variable(tf.random_normal([800]), name="bias9")
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([100,1]), name="weight10")
b2 = tf.Variable(tf.random_normal([1]), name="bias10")
hypothesis = tf.sigmoid(tf.matmul(L1,W2)+b2)


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) 



# luanch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})

        if step % 100 == 0:
            print(step, cost_val)


    # Accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis :", h, "\nCorrect :", c, "\nAccuracy :", a)
