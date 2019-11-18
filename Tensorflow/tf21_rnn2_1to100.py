import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

a = np.array(range(1,101))
print(len(a))

size = 12

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

x_train = dataset[:, 0:6]
y_train = dataset[:, 6:7]
print("x_train shape : ", x_train.shape)    # (89,6)
print("y_train shape : ", y_train.shape)    # (89,1)
print(x_train)
print(y_train)

x_train = np.reshape(x_train, (89,6,1))

x_test = np.array([[101,102,103,104,105,106],[102,103,104,105,106,107],[103,104,105,106,107,108],[104,105,106,107,108,109],[105,106,107,108,109,110]])
y_test = np.array([[107],[108],[109],[110],[111]])
x_test = np.reshape(x_test, (5,6,1))

print(x_test.shape)
print(y_test.shape)


# 2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(6,1), activation="relu"))
model.add(BatchNormalization())

model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1))


# 3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="loss", patience=20, mode="auto")
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, callbacks=[early_stopping])


# 4. 평가
# loss, acc = model.evaluate(x_test, y_test)
mse, _ = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
# print("loss : ", loss)
# print("acc : ", acc)
print("mse :", mse)
print("y_predict :\n", y_predict)



# 5. RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))



# 6. R2
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)