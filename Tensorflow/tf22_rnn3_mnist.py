import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

lr = 0.001
total_epochs = 30
batch_size = 128


n_input = 28
n_step = 28
n_hidden = 128
n_class = 10


## 모델구성
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

print(X)
print(Y)
print(W)
print(b)


## RNN에 학습에 사용할 셀 사용
## BasicRNNCell, BasicLSTMCell, GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(outputs)  # Tensor("rnn/transpose_1:0", shape=(?, 28, 128), dtype=float32)


## 결과를 Y의 형식으로 바꿔야함
## Y : [batch_size, n_class]
## outputs : [batch_size, n_step, n_hidden]
#            [n_step, batch_size, n_hidden]
#            [batch_size, n_hidden]

outputs = tf.transpose(outputs, [1,0,2])
print(outputs)  # Tensor("transpose:0", shape=(28, ?, 128), dtype=float32)
outputs = outputs[-1]
print(outputs)  # Tensor("strided_slice:0", shape=(?, 128), dtype=float32)


model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)



# 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epochs):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys})
        total_cost += cost_val

    print("Epochs :", "%04d" % (epoch + 1), "Avg. cost =", "{:.3f}".format(total_cost/total_batch))

print("최적화 완료")



# 결과확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print("정확도 >>", sess.run(accuracy, feed_dict={X:test_xs, Y:test_ys}))
        