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
x_train = dataset[:, 0:4]
y_train = dataset[:, 4]
print(x_train.shape)    # (6,4)
print(y_train.shape)    # (6, )
# print(x_train)
# print(y_train)

x_test = np.array([ [[11],[12],[13],[14]], [[12],[13],[14],[15]], [[13],[14],[15],[16]], [[14],[15],[16],[17]] ])
y_test = np.array([15,16,17,18])


## 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_sc = scaler.transform(x_train)
print("x_train >> ",x_train)
print("x_train_sc >> ", x_train_sc)

## 훈련데이터셋 차원변경
x_train_sc = np.reshape(x_train_sc, (6,4,1))
print("========== x_train_sc shape ==========\n", x_train_sc.shape)    # (6, 4, 1) >> 열의 데이터를 하나씩 짜르기
print("========== x_train_sc==========\n", x_train_sc)


x_test = np.reshape(x_test, (4,4))
x_test_sc = scaler.transform(x_test)
print("\n========== x_test ==========\n", x_test)
print("========== x_test_sc ==========\n", x_test_sc)

## 시험데이터셋 차원변경
x_test_sc = np.reshape(x_test_sc, (4,4,1))
print("========== x_test_sc shape ==========\n", x_test.shape) # (4, 4, 1)
y_test = y_test.shape # (4, )


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
model.fit(x_train_sc, y_train, epochs=10000, batch_size=1, verbose=1, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test_sc, y_test)

y_predict = model.predict(x_test_sc)
print("loss : ", loss)
print("acc : ", acc)
print("y_predict :", y_predict)

'''