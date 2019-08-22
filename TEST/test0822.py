import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data load
df = pd.read_csv("C:/Study/ML/Data/test0822.csv", sep=",", encoding="utf-8")

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
# print(df.head(5))
df = df.as_matrix()

## Preprocessing
df = df.astype("float32")
sc = StandardScaler()
df = sc.fit_transform(df)
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)


# print(df.shape) # (5479,8)
X = df[:3118,:]   # (1991-01-01 ~ 2007-07-15)
# print(X.shape)  # (3113,8)
Y = df[1:3119,:]  # (1991-01-02 ~ 2007-07-16)
# print(Y.shape)  # (3113,8)

## dataset split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

x_train = np.reshape(x_train, (2494,8,1))
x_test = np.reshape(x_test, (624,8,1))
print("x_train shape >> ", x_train.shape)   # (2494, 8, 1)
print("y_train shape >> ", y_train.shape)   # (2494, 8)
print("x_test shape >> ", x_test.shape)     # (624, 8, 1)   (2005-10-30 ~ 2007-07-15)
print("y_test shape >> ", y_test.shape)     # (624, 8)



# Model (stateful_lstm)
model = Sequential()
# model.add(LSTM(32, batch_input_shape=(1,8,1), stateful=True, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(8, batch_input_shape=(1,8,1), stateful=True, return_sequences=True))
# model.add(Dropout(0.5))
model.add(LSTM(4, batch_input_shape=(1,8,1), stateful=True))
model.add(Dropout(0.5))

model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(8))

model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["mse"])

num_epochs = 1
for epoch_idx in range(num_epochs):
    print("epoch :" + str(epoch_idx))
    ## num_epoch(50) * epochs(1) = 50회 반복
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, validation_data=(x_test, y_test))
    model.reset_states()

# ## Evaluate
# mse, _ = model.evaluate(x_train, y_train, batch_size=1)
# print("mse >>", mse)
# model.reset_states()

## Predict
y_prd = model.predict(x_test)

## Invert
y_prd = sc.inverse_transform(y_prd)
y_test = sc.inverse_transform(y_test)
print("y_predict >>", y_prd)


##########################################################################################################
# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_prd))


# R2
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_prd)
print("R2 : ", r2_y_predict)


