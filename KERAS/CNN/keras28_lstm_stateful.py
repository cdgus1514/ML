###########################################################################################
# stateful >> 상태를 유지하며 여러번 실행 가능
#
#
###########################################################################################
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping



# 1. 데이터셋 만들기
a = np.array(range(1,101))
Batch_size = 1
size = 5


## 연속적인 데이터셋 만들기
def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    
    print(type(aaa))

    return np.array(aaa)

dataset = split_5(a, size)
print("----------------------------------------------")

# print(dataset)
print("dataset.shape", dataset.shape) #(96,5)


## 학습, 시험데이터셋 생성 및 차원변환
x_train = dataset[:, 0:4]   # 1,2,3,4 ...
y_train = dataset[:, 4]     # 5,6,7,8 ...

x_train = np.reshape(x_train, (len(x_train), size-1, Batch_size))

x_test = x_train + 100      # 101, 102 ... 200
y_test = y_train + 100      # 105, 106 ... 200

print("x_train", x_train.shape) # (96, 4, 1)
print("y_train", y_train.shape) # (96, )
print("x_test", x_test.shape)   # (96, 4, 1)
print("y_test", y_test.shape)   # (96, )



# 2. 모델구성
model = Sequential()
model.add(LSTM(15, batch_input_shape=(Batch_size,4,1), stateful=True))
# input_shape >> batch_input_shape : (배치사이즈, 4열, 1개씩) // 상태유지


# model.add(Dense(4096))
# model.add(Dense(2048))
# model.add(Dense(1024))
model.add(Dense(10))
model.add(Dense(35))
model.add(Dense(25))
model.add(Dense(10))

model.add(Dense(1))
## 0.2
# model.summary()
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

## Tensorboard
tb_hist = keras.callbacks.TensorBoard(log_dir="./graph", histogram_freq=0, write_graph=True, write_images=True)

## EarlyStopping 적용
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)    # 변화값이 patience이상 변경 없을경우 중지

## fit 구현 >> 실행 후 섞인 데이터를 그대로 유지한채 50번 반복실행
num_epochs = 50
for epoch_idx in range(num_epochs):
    print("epoch : " +str(epoch_idx))
    ## num_epoch(50) * epochs(1) = 50회 반복 
    model.fit(x_train, y_train, epochs=1, batch_size=Batch_size, verbose=2, shuffle=False, validation_data=(x_test, y_test), callbacks=[early_stopping_callback])
    ## 상태유지를 적용하는 경우에는 실행, 예측 후 현재상태를 리셋해야함 >> 값 변경 X
    model.reset_states()



mse, _ = model.evaluate(x_train, y_train, batch_size=Batch_size)
print("mse : ", mse)
model.reset_states()



# 3. 예측
y_predict = model.predict(x_test, batch_size=Batch_size)
print(y_predict[0:5])   # 105, 106, 107, 108, 109



# 4. RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):    # 결과값 :? 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("\nRMSE : ", RMSE(y_test, y_predict))



# 5. R2
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("\nR2 : ", r2_y_predict)









# mse >> 1 이하로 (hidden layer 3개이상 추가, dropout | batchnormalization 적용)
# RMSE함수 적용
# R2함수 적용
# early_stopping 적용
# tensorboard 적용
# matplotlib 적용 (mse/epochs