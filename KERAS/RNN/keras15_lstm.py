import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1,11))
size = 5

# 1. 연속적인 데이터셋 만들기
def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    
    print(type(aaa))

    return np.array(aaa)

dataset = split_5(a, size)

print("====================")
## 훈련데이터셋 생성(n,4), (n,)
x_train = dataset[:, 0:4]
y_train = dataset[:, 4]
print(x_train.shape)    # (6,4)
print(y_train.shape)    # (6,)
# print(x_train)
# print(y_train)

# print("debug >> ", len(a)-size+1) # 6
# x_train = np.reshape(x_train, (6,4,1))
x_train = np.reshape(x_train, (len(a)-size+1, 4, 1))
print(x_train.shape)    # (6,4,1)
print(y_train.shape)    # (6,)

print("====================")
## 시험데이터셋 생성
x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]], [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
y_test = np.array([15,16,17,18])

print(x_test.shape) # (4,4,1)
print(y_test.shape) # (4, )

'''
# 2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(5))

model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1))

# model.summary()

# 3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="loss", patience=30, mode="auto")
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print("loss : ", loss)
print("acc : ", acc)
print("y_predict :", y_predict)

'''