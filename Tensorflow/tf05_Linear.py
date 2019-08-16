import tensorflow as tf

tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1], name="weight"))
b = tf.Variable(tf.random_normal([1], name="bias"))

# y = xw+b 그래프
hypothesis = x_train * W + b


# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss = "mse"

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)    # optimizer="GradientDescent"


# 실행
with tf.Session() as sess:
    ## 변수 초기화(필수)
    sess.run(tf.global_variables_initializer())

    ## fit
    for step in range(2001):    # epochs
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])   # fit

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)