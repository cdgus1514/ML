import tensorflow as tf
import numpy as np

# data
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.469971, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape) # (8,4)
print(y_data.shape) # (8,1)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = StandardScaler()
x_data = sc.fit_transform(x_data)


X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# model
W = tf.Variable(tf.random_normal([4,1], name="weight"))
b = tf.Variable(tf.random_normal([1], name="bias"))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)

predicted = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y_data, 1))
# accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))


# launch graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(301):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
    print(step, "Cost :", cost_val, "\nPrediction :", hy_val)




from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_data, sess.run(hypothesis, feed_dict={X:x_data})))


from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_data, sess.run(hypothesis, feed_dict={X:x_data}))
print("R2 : ", r2_y_predict)