import numpy as np

## 1. 학습데이터
x = np.array([range(1000), range(3110,4110), range(1000)])
y = np.array([range(5010, 6010)])
print(x.shape)
print(y.shape)

# 차원변경
x = np.transpose(x)
y = np.transpose(y)
print(x.shape)
print(y.shape)



# 데이터 분할( 6/2/2)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)




## 2. 모델구성(레이어, 노드 개수 설정)
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers


model = Sequential()
# model.add(Dense(100, input_shape=(3, ), activation="relu"))
model.add(Dense(1000, input_shape=(3, ), activation="relu", kernel_regularizer=regularizers.l1(0.05)))  # Lasso regression
model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(1000, kernel_regularizer=regularizers.l2(0.05)))
model.add(Dropout(0.8))
model.add(Dense(1000))
model.add(Dropout(0.8))
model.add(Dense(1000))
model.add(Dropout(0.8))
model.add(Dense(1000, kernel_regularizer=regularizers.l2(0.05)))
model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(1))




## 3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_val, y_val))




## 4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)




## RMSE 구하기 (오차비교)
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):    # 결과값 :? 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))




## R2 구하기 (결정계수 >> 1에 가까울수록 좋음)
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

print("loss : ", loss)

