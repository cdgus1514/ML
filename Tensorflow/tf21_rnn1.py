import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
tf.set_random_seed(777)

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1,1)
print(_data.shape)  # (7,1)
print(_data)
print(_data.dtype)


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
_data = enc.fit_transform(_data).toarray().astype("float32")    # float64로 변경되서 32로 형변환

print(_data)
print(_data.shape)  # (7,5)
print(type(_data))
print(_data.dtype)


x_data = _data[:6,]
y_data = _data[1:,]
y_data = np.argmax(y_data, axis=1)

print(x_data)
print(y_data)

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)

print(x_data.shape) # (1,6,5)
print(x_data.dtype)
print(y_data.shape) # (1,6)



## 데이터 구성
### x : (batch_size, sequence_length, input_dim) = 1,6,5
### 첫번째 아웃풋 : hidden_size = 2
### 첫번째 결과 : 1,6,5
num_classes = 5
batch_size = 1      # 전체행
sequence_length = 6 # column
input_dim = 5       # 몇개씩 작업
hidden_size = 5     # 첫번째 노드 출력 객수
lr = 0.1


X = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])
print(X)
print(Y)



## 모델 구성
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)    # output
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)    # (5,(1,6,5))
print(outputs)
print(outputs.shape)    # (1,6,5)


# ## FC layer
# X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# print(X_for_fc) # Tensor("Reshape:0", shape=(6, 5), dtype=float32)
# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)    # cost
train = tf.compat.v1.train.AdamOptimizer(0.1).minimize(loss)    # optimizer

prediction = tf.argmax(outputs, axis=2) # one-hot encoding >> original
print(prediction.shape) # (1,6)


'''
## 모델 실행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i, "loss :", l, "prediction :", result, "true Y :", y_data)
        # print(sess.run(weights))

        results_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str :", "".join(results_str))

'''