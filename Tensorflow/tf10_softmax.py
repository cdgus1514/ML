import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]


X = tf.placeholder(tf.float32, shape=[None,4])
Y = tf.placeholder(tf.float32, shape=[None,3])
nb_calsses = 3

W = tf.Variable(tf.random_normal([4,nb_calsses]), name="weight")
b = tf.Variable(tf.random_normal([nb_calsses]), name="bias")


hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

# model compile
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1)) # loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # optimizer


# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, cost_val)


    # Testing one-hot encoding
    print("=======================")
    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))

    
    print("=======================")
    b = sess.run(hypothesis, feed_dict={X:[[1,3,4,3]]})
    print(b, sess.run(tf.argmax(b,1)))

    print("=======================")
    c = sess.run(hypothesis, feed_dict={X:[[1,1,0,1]]})
    print(c, sess.run(tf.argmax(c,1)))

    print("=======================")
    all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print(all, sess.run(tf.argmax(all,1)))



'''
[[1.3890490e-03 9.9860185e-01 9.0612921e-06]] [1]
=======================
[[0.9311919  0.06290223 0.00590592]] [0]
=======================
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
=======================
[[1.3890478e-03 9.9860197e-01 9.0612930e-06]
 [9.3119192e-01 6.2902197e-02 5.9059118e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]
 [9.3119192e-01 6.2902197e-02 5.9059118e-03]] [1 0 2]
'''