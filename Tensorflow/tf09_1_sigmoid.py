import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


hypothesis = tf.sigmoid(tf.matmul(X,W)+b)


# model compile
## 로지스틱 레그레이션 sigmoid >> cost = - 사용
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 0.5보다 크면 1, 작으면 0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    # 이진분류 모델에서만 사용 가능

# launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, "Cost: ", cost_val)

    
    ## Accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("Hypothesis :\n", h, "\nCorrect (Y) :\n", c, "\nAccuracy :", a)



'''
Hypothesis :
[[0.03074032]
 [0.15884677]
 [0.3048674 ]
 [0.78138196]
 [0.93957496]
 [0.9801688 ]]
Correct (Y) :
[[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]]
Accuracy : 1.0
'''