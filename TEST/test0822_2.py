import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data load
df = pd.read_csv("C:/Study/ML/Data/test0822.csv", sep=",", encoding="utf-8")
# df = pd.read_csv("/content/test0822.csv", sep=",", encoding="utf-8")

print(df.head(5))
## 데이터 확인
#          date  kp_0h  kp_3h  kp_6h  kp_9h  kp_12h  kp_15h  kp_18h  kp_21h
# 0  1999-01-01    0.0    2.0    1.0    2.0     2.0     1.0     1.0     1.0
# 1  1999-01-02    1.0    2.0    2.0    3.0     3.0     2.0     2.0     1.0
# 2  1999-01-03    2.0    2.0    0.0    0.0     1.0     1.0     1.0     1.0
# 3  1999-01-04    1.0    2.0    3.0    2.0     3.0     2.0     1.0     2.0
# 4  1999-01-05    3.0    3.0    2.0    3.0     1.0     1.0     2.0     1.0
print("================================================")
df = df.drop(["date"], axis=1)
print(df.head(5))
df = df.as_matrix()

## Preprocessing
df = df.astype("float32")
sc = StandardScaler()
df = sc.fit_transform(df)

# print(df.shape) # (5479,8)
X = df[:,:]
Y = df[:,:]

dataX = [] # 7일치
dataY = [] # 7일 종가
for i in range(0, len(Y) - 5):
    _x = X[i : i+5]
    _y = Y[i + 5] # 다음 나타날 주가(정답)
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

X = np.array(dataX)
Y = np.array(dataY)


print(X.shape) # (5474,5,8)
print(Y.shape) # (5474,8)

# print(X[3108]) # (2007-07-10까지)
# print(X[3118]) # (2007-07-16부터)


x_train = X[:3108, :]
y_train = Y[:3108, :]
# print(x_train[0])
# print()
# print(y_train[0])

x_test = X[3118:, :]
y_test = Y[3118:, :]
# print(x_test[0])
# print()
# print(y_test[0])

x_train, x_test2, y_train, y_test2 = train_test_split(x_train, y_train, test_size=0.3, shuffle=False)


# x_train = np.reshape(x_train, (3108,5,8,1))
# x_test = np.reshape(x_test, (2356,5,8,1))
print("x_train shape >> ", x_train.shape)   # (2175, 5, 8)
print("y_train shape >> ", y_train.shape)   # (2175, 8)
print("x_test shape >> ", x_test.shape)     # (2356, 5, 8)
print("y_test shape >> ", y_test.shape)     # (2356, 8)

print("x_test2 shape >> ", x_test2.shape)   # (933, 5, 8)
print("y_test2 shape >> ", y_test2.shape)   # (933, 8)

# print(x_test2[932])
# print(y_test2[932])


# Model (stateful_lstm)
model = Sequential()
model.add(LSTM(32, activation="relu", input_shape=(5,8), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation="relu", return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(16, activation="relu"))

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8))

model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["mse"])

model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(x_test, y_test))

## Predict
y_prd = model.predict(x_test2, batch_size=8)

## Invert
y_prd = sc.inverse_transform(y_prd)
y_test = sc.inverse_transform(y_test)
y_test2 = sc.inverse_transform(y_test2)

y_prd = y_prd.astype("int32")
print("y_predict >>", y_prd[925:])


##########################################################################################################
# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test2, y_predict))

print("RMSE : ", RMSE(y_test, y_prd))


# R2
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test2, y_prd)
print("R2 : ", r2_y_predict)

