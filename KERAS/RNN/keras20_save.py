## 2. 모델구성(레이어, 노드 개수 설정)
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers


model = Sequential()
model.add(Dense(10, input_shape=(3, ), activation="relu", kernel_regularizer=regularizers.l1(0.01)))  # Lasso regression
model.add(Dense(100))
model.add(Dense(10))
# model.add(Dense(10000, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(100))
model.add(Dense(1))


model.save("savetest01.h5")
print("save complete")