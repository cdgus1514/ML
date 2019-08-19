import tensorflow as tf

tf.set_random_seed(777)

# 데이터 input=3, output=1
x_data = [[73,80,75],[96,88,93],[89,91,90],[96,98,100],[73,66,70]]

y_data = [[152],[185],[180],[196],[142]]


X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name="weight") # [input, output]
b = tf.Variable(tf.random_normal([1]), name="bias")     # [output]


hypothesis = tf.matmul(X,W)+b


# model compile
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)



# launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction : \n", hy_val)
            # y_data = [[152],[185],[180],[196],[142]]
            




'''
2000 Cost:  2.8760915
Prediction :
 [[154.08934]
 [185.89127]
 [181.05614]
 [193.40874]
 [140.82072]]
'''