import tensorflow as tf
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, LSTM
# from keras.utils import np_utils
from keras.utils import to_categorical

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1,1)
print(_data.shape)  # (7,1)
print(_data)
print(_data.dtype)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
_data = enc.fit_transform(_data).toarray().astype("float32")    # float64로 변경되서 32로 형변환
# _data = to_categorical(_data)


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


model = Sequential()
model.add(LSTM(512, input_shape=(6,5), activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dense(6, activation="relu"))


model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.fit(x_data, y_data, epochs=250, batch_size=batch_size)


loss, acc = model.evaluate(x_data, y_data)


y_predict = model.predict(x_data)
y_predict = y_predict.astype("int32")   # (1,6)


print(y_predict)

results_str = [idx2char[c] for c in np.squeeze(y_predict)]
print("\nPrediction str :", "".join(results_str))