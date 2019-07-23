import numpy as np

# 1. 학습데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])  #10행 1열
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x3 = np.array([101, 102, 103, 104, 105, 106])   #6행 1열
x4 = np.array(range(30, 50))


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


# 2. 모델구성(레이어, 노드 개수 설정)
model.add(Dense(7, input_dim=1, activation="relu")) #  input_dim=1 >> (column이 1개인 input), relu(완전 열결 층)
# model.add(Dense(5, input_shape=(1, ), activation="relu")) # input_shape=(1, ) >>(1행 n열인 input)
# model.add(Dense(13))
# model.add(Dense(8))
# model.add(Dense(3))
# model.add(Dense(1))

model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(30))
model.add(Dense(8000))
model.add(Dense(1))


# 3. 훈련
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
# 훈련실행(구성한 모델에 x,y 데이터를 n개씩 짤라서 n번 반복 훈련)
# model.fit(x, y, epochs=20, batch_size=3)   # epochs >> 만들어준 모델링을 n회 반복
                                           # batch_size >> n개씩 짤라서 연산
model.fit(x_train, y_train, epochs=100, batch_size=1)


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