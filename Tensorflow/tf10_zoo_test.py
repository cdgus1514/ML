import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt("C:/Study/ML/Data/data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]
print(x_data.shape) # (101,16)
print(y_data.shape) # (101,1)


# ## one hot encoding
# from keras.utils import np_utils
# y_data = np_utils.to_categorical(y_data)
y_one_hot = tf.one_hot(y_data, depth=7).eval(session=tf.Session())
print(y_one_hot.shape)

y_one_hot = np.reshape(y_one_hot, (101,7))

# print(y_data)
print(x_data.shape)     # (101,16)
print(y_one_hot.shape)  # (101,7)

X = tf.placeholder(tf.float32, shape=[None,16])
Y = tf.placeholder(tf.float32, shape=[None,7])


# model
W = tf.Variable(tf.random_normal([16,7]), name="weight")
b = tf.Variable(tf.random_normal([7]), name="bias")
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

# model compile
## loss="categorical_crossentropy"
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# predicted = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
predicted = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))



# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_one_hot})

        if step % 100 == 0:
            print(step, cost_val)


    # Acuuracy
    h, a, l = sess.run([hypothesis, accuracy, cost], feed_dict={X:x_data, Y:y_one_hot})
    print("Hypothesis :\n", h, "\n\nacc :", a, "\nloss :", l)
    


'''
acc : 1.0
loss : 0.06430749
'''