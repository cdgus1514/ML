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

L1 = tf.layers.dense(X, 256, activation=tf.nn.relu)
L1 = tf.layers.dropout(L1, 0.2)

L1 = tf.layers.dense(X, 512, activation=tf.nn.relu)
L1 = tf.layers.dropout(L1, 0.2)

L1 = tf.layers.dense(X, 256, activation=tf.nn.relu)
L1 = tf.layers.dropout(L1, 0.2)

L1 = tf.layers.dense(X, 512, activation=tf.nn.relu)
L1 = tf.layers.dropout(L1, 0.2)

L2 = tf.layers.dense(L1, 64, activation=tf.nn.relu)

logits = tf.layers.dense(L2, 3, activation=None)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
train = tf.train.AdamOptimizer(0.01).minimize(cost)

## Test model
is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))
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
