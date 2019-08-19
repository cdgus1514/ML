import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]

# set wrong weight
W = tf.Variable(5.0)


# linear model
hypothesis = X * W

# model compile
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, W_val = sess.run([train, W])
        print(step, W_val)


'''
0 5.0
1 1.2666664
2 1.0177778
3 1.0011852
4 1.000079
5 1.0000052

95 1.0
96 1.0
97 1.0
98 1.0
99 1.0
100 1.0

잘못된 weight값을 줘도 훈련하면서 교정됨
'''