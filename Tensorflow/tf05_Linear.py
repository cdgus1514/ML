import tensorflow as tf

tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1], name="weight"))
b = tf.Variable(tf.random_normal([1], name="bias"))

# model(graph) >> y = xw+b
hypothesis = x_train * W + b

# model compile

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss = "mse"
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # optimizer


# luanch graph
with tf.Session() as sess:
    ## initializer
    sess.run(tf.global_variables_initializer())

    ## model.fit >> ssesion.run
    for step in range(2001):  ## epochs
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
