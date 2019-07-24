import numpy as np

# 1. 학습데이터
x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])
x2 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])
# print(x.shape)
# print(y.shape)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)



# 데이터 분할( 6/2/2)
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size=0.4)   # traint 60 / test 40
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=66, test_size=0.5) # val 20 / test 20

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size=0.4)   # traint 60 / test 40
x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=66, test_size=0.5) # val 20 / test 20

# print("x_test")
# print(x_test, len(x_test))
# print("\nx_train")
# print(x_train, len(x_train))
# print("\nx_val")
# print(x_val, len(x_val))

# print(x1_train.shape)
# print(x2_train.shape)
# print(x1_test.shape)
# print(x1_test.shape)
# print(x1_val.shape)
# print(x2_val.shape)



# 2. 모델구성(레이어, 노드 개수 설정)
from keras.models import Sequential, Model
from keras.layers import Dense, Input
model = Sequential()

## 함수형 모델 (앙상블 모델)
input1 = Input(shape=(3,))                      # 첫번째 인풋 레이어에 shape=3
dense1 = Dense(100, activation="relu")(input1)  # input 300 >> output 100
dense1_2 = Dense(30)(dense1)                    # input 100 >> output 30
dense1_3 = Dense(10)(dense1_2)                  # input 30  >> output 10

input2 = Input(shape=(3,))
dense2 = Dense(50, activation="relu")(input2)


## 순차적 모델(하나의 모델만 사용가능)
# model.add(Dense(5, input_shape=(3, ), activation="relu"))
# model.add(Dense(3000))
# model.add(Dense(30))
# model.add(Dense(3000))
# model.add(Dense(30))
# model.add(Dense(3000))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(3000))
# model.add(Dense(5))
# model.add(Dense(3))


# 3. 훈련
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
# 훈련실행(구성한 모델에 x,y 데이터를 n개씩 짤라서 n번 반복 훈련)
# model.fit(x, y, epochs=20, batch_size=3)   # epochs >> 만들어준 모델링을 n회 반복
                                           # batch_size >> n개씩 짤라서 연산
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))


# 4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("acc : ", acc)

# y값 예측 (x값 >> 훈련시킨 값, x2값 >> 훈련시킨 모델에서 나온 w값으로 새로운 데이터 결과값 예측)
# acc(분류모델용, 근사값을 이용해 분류), predict(acc가 100%이어도 100% 정확하게 예측값이 나오지는 않음)
y_predict = model.predict(x_test)
print(y_predict)


# RMSE 구하기 (오차비교)
# x_test값 + y_predict >> 제곱한 값들의 평균에 루트 // 낮을수록 정확
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):    # 결과값 :? 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기 (결정계수 >> 1에 가까울수록 좋음)
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

