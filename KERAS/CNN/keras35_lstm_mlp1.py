# input_shape(4,1)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

a = np.array(range(1,101))
print(len(a))

size = 8

## 연속적인 데이터셋 만들기
def split_44(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    
    # print(type(aaa))
    # print(aaa)

    return np.array(aaa)

dataset = split_44(a, size)

print("====================")
x_train = dataset[:, 0:4]
y_train = dataset[:, 4:8]
print("x_train shape : ", x_train.shape)    # (93,4)
print("y_train shape : ", y_train.shape)    # (93,4)
print(x_train)
print(y_train)


x_train = np.reshape(x_train, (len(a)-size+1, 4, 1))    
print("x_train reshape : ", x_train.shape)   # (93,4,1)
print("y_train shape : ", y_train.shape)     # (93,4)



## 시험데이터셋 구성
x_test = np.array([[101,102,103,104], [102,103,104,105], [103,104,105,106], [104,105,106,107]])
y_test = np.array([[105,106,107,108], [106,107,108,109], [107,108,109,110], [108,109,110,111]])
print("x_test shape : ", x_test.shape) # (4,4,1)
print("y_test.shape : ", y_test.shape) # (4, )

x_test = np.reshape(x_test, (4,4,1))
print("x_test reshape : ", x_test.shape)

# 2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(5))
model.add(BatchNormalization)

model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(4))


# 3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="loss", patience=10, mode="auto")
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print("loss : ", loss)
print("acc : ", acc)
print("y_predict :", y_predict)



# 5. RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):    # 결과값 :? 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))



# 6. R2
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)