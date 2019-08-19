import tensorflow as tf

tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1], name="weight"))
b = tf.Variable(tf.random_normal([1], name="bias"))

# 모델구성 (y = xw+b 그래프)
hypothesis = x_train * W + b


# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss = "mse"


# optimizer >> "GradientDescent"
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# 실행
with tf.Session() as sess:
    ## 변수 초기화(필수)
    sess.run(tf.global_variables_initializer())

    ## fit (# model.fit >> ssesion.run)
    for step in range(2001):  # epochs
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
