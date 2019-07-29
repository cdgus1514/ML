import numpy as np

# 1. 학습데이터
x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])

x2 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

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
# print(x1_test, len(x1_test))
# print("\nx_train")
# print(x1_train, len(x1_train))
# print("\nx_val")
# print(x1_val, len(x1_val))

# print(x1_train.shape)
# print(x2_train.shape)
# print(x1_test.shape)
# print(x1_test.shape)
# print(x1_val.shape)
# print(x2_val.shape)



# 2. 모델구성(레이어, 노드 개수 설정)
from keras.models import Sequential, Model
from keras.layers import Dense, Input

## 함수형 모델 (앙상블 모델)
input1 = Input(shape=(3,))                      # 첫번째 인풋 레이어에 shape=3
dense1 = Dense(100, activation="relu")(input1)  # input 300 >> output 100
dense1_2 = Dense(30)(dense1)                    # input 100 >> output 30
dense1_3 = Dense(10)(dense1_2)                  # input 30  >> output 10

input2 = Input(shape=(3,))
dense2 = Dense(50, activation="relu")(input2)
dense2_2 = Dense(7)(dense2)

## 모델을 하나로 합치기
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])

mid1 = Dense(10)(merge1)
mid2 = Dense(5)(mid1)
mid3 = Dense(7)(mid2)

### output 모델 구성
output1 = Dense(30)(mid3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(mid3)
output2_2 = Dense(7)(output2)
output2_3 = Dense(3)(output2_2)


model = Model(input=[input1, input2], output=[output1_3, output2_3])
# model.summary()


# 3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y2_val]))



# 4. 평가예측
# loss, acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print("acc : ", acc)


y1_predict, y2_predict = model.predict([x1_test, x2_test])
print(y1_predict, y2_predict)



# RMSE 구하기 (오차비교)
# x_test값 + y_predict >> 제곱한 값들의 평균에 루트 // 낮을수록 정확
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):    # 결과값 :? 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("y1 RMSE : ", RMSE(y1_test, y1_predict))
print("y1 RMSE : ", RMSE(y2_test, y2_predict))



# R2 구하기 (결정계수 >> 1에 가까울수록 좋음)
from sklearn.metrics import r2_score

r2_y1_predict = r2_score(y1_test, y1_predict)
print("y1 R2 : ", r2_y1_predict)

r2_y2_predict = r2_score(y2_test, y2_predict)
print("y2 R2 : ", r2_y2_predict)