import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.load("iris_train.npz")
x_train = xy["x_train"]
y_train = xy["y_train"]

xy = np.load("iris_test.npz")
x_test = xy["x_test"]
y_test = xy["y_test"]

print(x_train.shape, y_train.shape) # (120,4) (120,3)
print(x_test.shape, y_test.shape)   # (30,4) (30,3)



# Graph
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

## 변수구성하면서 초기화 진행
W1 = tf.get_variable("W1", shape=[4, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.leaky_relu(tf.matmul(X,W1)+b1)
# L1 = tf.nn.dropout(L1,0.5)

# W2 = tf.Variable(tf.random_normal([256,512]))
W2 = tf.get_variable("W2", shape=[256,512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.leaky_relu(tf.matmul(L1,W2)+b2)
# L2 = tf.nn.dropout(L2,0.5)

# W3 = tf.Variable(tf.random_normal([512,10]))
W3 = tf.get_variable("W3", shape=[512,3], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([3]))
# hypothesis = tf.nn.softmax(tf.matmul(L2,W3)+b3)
hypothesis = tf.matmul(L2,W3)+b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
train = tf.train.AdamOptimizer(0.01).minimize(cost)

## Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))



# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_train, Y:y_train})

        if step % 10 == 0:
            print(step, cost_val)

    
    # Test the model using test sets
    print("Accuracy :", accuracy.eval(session=sess, feed_dict={X:x_test, Y:y_test}))