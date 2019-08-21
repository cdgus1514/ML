import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# Data load
xy = np.loadtxt("C:/Study/ML/Data/data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]
print(x_data.shape) # (101,16)
print(y_data.shape) # (101,1)



# Graph
X = tf.placeholder(tf.float32, shape=[None,16])
Y = tf.placeholder(tf.int32, shape=[None,1])

## one-hot encoding
Y_one_hot = tf.one_hot(Y, 7)
print("one-hot >>", Y_one_hot) # one-hot >> Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])
print("rehsape one-hot >>", Y_one_hot) # rehsape one-hot >> Tensor("Reshape:0", shape=(?, 7), dtype=float32)

## layer
W = tf.Variable(tf.random_normal([16,7]), name="weight")
b = tf.Variable(tf.random_normal([7]), name="bias")
logists = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logists)

# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logists, labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # predict와  Y 비교
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X:x_data, Y:y_data})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))


    ## predict
    pred = sess.run(prediction, feed_dict={X:x_data})
    ## y_data(N,1) >> flatten() >> (N,)
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] prediction: {} True Y: {}".format(p==int(y), p, int(y)))



'''
Step:     0     Cost: 5.480     Acc: 37.62%
Step:   100     Cost: 0.806     Acc: 79.21%
Step:   200     Cost: 0.488     Acc: 88.12%
Step:   300     Cost: 0.350     Acc: 90.10%
Step:   400     Cost: 0.272     Acc: 94.06%
Step:   500     Cost: 0.222     Acc: 95.05%
Step:   600     Cost: 0.187     Acc: 97.03%
Step:   700     Cost: 0.161     Acc: 97.03%
Step:   800     Cost: 0.141     Acc: 97.03%
Step:   900     Cost: 0.124     Acc: 97.03%
Step:  1000     Cost: 0.111     Acc: 97.03%
Step:  1100     Cost: 0.101     Acc: 99.01%
Step:  1200     Cost: 0.092     Acc: 100.00%
Step:  1300     Cost: 0.084     Acc: 100.00%
Step:  1400     Cost: 0.078     Acc: 100.00%
Step:  1500     Cost: 0.072     Acc: 100.00%
Step:  1600     Cost: 0.068     Acc: 100.00%
Step:  1700     Cost: 0.064     Acc: 100.00%
Step:  1800     Cost: 0.060     Acc: 100.00%
Step:  1900     Cost: 0.057     Acc: 100.00%
Step:  2000     Cost: 0.054     Acc: 100.00%
'''