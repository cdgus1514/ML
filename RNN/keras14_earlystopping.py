import numpy as np

# 1. 학습데이터
x = np.array([range(100), range(311,411), range(601,701)]).reshape(100,3)
y = np.array([range(501, 601), range(711, 811), range(901, 1001)]).reshape(100,3)

print(x.shape)
print(y.shape)

# x = np.transpose(x)
# y = np.transpose(y)


# 데이터 분할( 6/2/2)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)   # traint 60 / test 40
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5) # val 20 / test 20

print(x_train.shape)



from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


# 2. 모델구성(레이어, 노드 개수 설정)
model.add(Dense(5, input_shape=(3, ), activation="relu")) # input_shape=(1, ) >>(1행 n열인 input)
model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3000))
model.add(Dense(5))
model.add(Dense(3))


# 3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="loss", patience=100, mode="auto")
hist = model.fit(x_train, y_train, epochs=10000, verbose=1, callbacks=[early_stopping])
# hits = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))


# 4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)


y_predict = model.predict(x_test)
print(y_predict)


# RMSE 구하기 (오차비교)
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):    # 결과값 :? 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기 (결정계수 >> 1에 가까울수록 좋음)
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

