import tensorflow as tf

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]


W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

## 모델구성
hypothesis = x * W + b


## 모델 compile
cost = tf.reduce_sum(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)


# launch
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], {x: x_train, y: y_train})

    # evaluate training accuracy
    W_val, b_val, cost_val = sess.run([W,b,cost], feed_dict={x:x_train, y:y_train})
    print(f"W: {W_val} b: {b_val} cost: {cost_val}")

