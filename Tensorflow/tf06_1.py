import tensorflow as tf

tf.set_random_seed(777)


W = tf.Variable(tf.random_normal([1], name="weight"))
b = tf.Variable(tf.random_normal([1], name="bias"))

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

## 모델구성
hypothesis = X * W + b


## 모델 compile
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # loss = "mse"
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # optimizer = "GD"


# 실행
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})


        # if step % 20 == 0:
            # print(step, cost_val, W_val, b_val)


    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

