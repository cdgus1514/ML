import numpy as np

# 1. 학습데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]


from sklearn.model_selection import train_test_split
# 6/2/2
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)   # traint 60 / test 40
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5) # val 20 / test 20


# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=66, test_size=0.3)


print("x_test", x_test, len(x_test))
print()
print("x_train", x_train, len(x_train))
print()
print("x_val", x_val, len(x_val))


'''
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


# 2. 모델구성(레이어, 노드 개수 설정)
model.add(Dense(7, input_dim=1, activation="relu")) #  input_dim=1 >> (column이 1개인 input), relu(완전 열결 층)
# model.add(Dense(5, input_shape=(1, ), activation="relu")) # input_shape=(1, ) >>(1행 n열인 input)

model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(3000))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3000))
model.add(Dense(5))
model.add(Dense(1))


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
'''