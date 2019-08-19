import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt("C:/Study/ML/Data/data-01-test-score.csv", delimiter=",", dtype=np.float32)
# xy = np.loadtxt("./Data/data-01-test-score.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data, "\nx_data shape >>", x_data.shape)    # (25,3)
print(y_data, "\ny_data shape >>", y_data.shape)    # (25,1)


X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name="weight") # [input, output]
b = tf.Variable(tf.random_normal([1]), name="bias")     # [output]


# hypothesis = tf.matmul(X,W)+b
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)


# model compile
## 로지스틱 레그레이션에서 cost =
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction : \n", hy_val)
            




'''
2000 Cost:  24.722485
Prediction :
 [[154.42894 ]
 [185.5586  ]
 [182.90646 ]
 [198.08955 ]
 [142.52043 ]
 [103.551796]
 [146.79152 ]
 [106.70152 ]
 [172.15207 ]
 [157.13037 ]
 [142.5532  ]
 [140.17581 ]
 [159.59953 ]
 [147.35217 ]
 [187.26833 ]
 [153.3315  ]
 [175.3862  ]
 [181.3706  ]
 [162.1332  ]
 [172.44307 ]
 [173.06042 ]
 [164.7337  ]
 [158.24257 ]
 [192.79166 ]
'''