import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPool1D, Embedding, Input, Flatten
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


## 1. 데이터 불러오기 >> csv파일 업로드 필요
# df = pd.read_csv("/content/kospi200test.csv", encoding="utf-8")
df = pd.read_csv("/content/kospi200test.csv", sep=",", engine="python")

print("## 데이터 확인 ##")
print(df.head(5))
## 데이터 확인 ##
#          Date     Open    Hight      Low    Close  Volume    dall
# 0  2019-08-01  2024.00  2005.31  1987.12  2017.34  455515  1195.8
# 1  2019-07-31  2036.46  2041.16  2010.95  2024.55  589386  1183.1
# 2  2019-07-30  2035.32  2044.59  2032.61  2038.68  547029  1181.6
# 3  2019-07-29  2059.13  2063.13  2025.01  2029.48  608670  1183.5
# 4  2019-07-26  2063.35  2068.16  2054.64  2066.26  589074  1184.8

print("original shape : ", df.shape)  #(600,7)

print("================================================")

# dataset_temp = df.drop(['Date', 'dall'], axis=1)
# dataset_temp = df.drop(['Date', 'Volume', 'dall'], axis=1)
dataset_temp = df.drop(['Date', 'Close', 'Volume', 'dall'], axis=1)
dataset_temp = dataset_temp.as_matrix()


# 내림차순 정렬
new_dataset = []
col_count = (len(dataset_temp))

for i in range(col_count-1, -1, -1):
  new_dataset.append(dataset_temp[i])
  
new_dataset_temp = np.array(new_dataset)
print("new_dataset_temp shape : ", new_dataset_temp.shape)  # (600,4)


## 종가 데이터셋 분리
close = []
for i in range(col_count):
#   close.append(new_dataset_temp[i][3])
  close.append(new_dataset_temp[i][2])


close = np.array(close)
print("close shape : ", close.shape)
close = np.reshape(close, (col_count,1))
print("close reshape : ", close.shape)

print(new_dataset_temp)


## 데이터셋 스케일링
sc = MinMaxScaler()
dataset_sc = sc.fit_transform(new_dataset_temp)


# 학습, 시험데이터셋 분할
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(dataset_sc, close, shuffle=False, test_size=0.3)   # traint : test >> 7:3

# X_train = X_train.reshape(420,4,1)
X_train = X_train.reshape(420,3,1)
# X_test = X_test.reshape(180,4,1)
X_test = X_test.reshape(180,3,1)


print("\n### 최종 shape ###")
print("X_train shape >> ", X_train.shape, len(X_train))   # (420,4)
print("Y_train shape >> ", Y_train.shape, len(Y_train))   # (420,1)
print("X_test shape >> ", X_test.shape, len(X_test))      # (180,4)
print("Y_test shape >> ", Y_test.shape, len(Y_test))      # (180,1)


# 2. 모델
# input = Input(shape=(4,1))
input = Input(shape=(3,1))


# conv = Conv1D(4,1, padding="same")(input)
conv = Conv1D(3,1, padding="same", activation="relu")(input)
pool = MaxPool1D(2)(conv)
conv = Conv1D(3,1, padding="same", activation="relu")(pool)
# pool = MaxPool1D(2)(conv)

flat = Flatten()(pool)

dense = Dense(128)(flat)
dense = Dense(256)(dense)
dense = Dense(512)(dense)
dense = Dense(1024)(dense)
dense = Dense(128)(dense)
dense = Dense(64)(dense)
dense = Dense(1)(dense)

model = Model(input, dense)


model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["mse"])


## EarlyStopping 적용
# stop = EarlyStopping(monitor="val_loss", patience=50, mode="auto")

# model.fit(X_train, Y_train, epochs=1000, batch_size=15, validation_data=(X_test, Y_test), callbacks=[stop])
model.fit(X_train, Y_train, epochs=1000, batch_size=4, validation_data=(X_test, Y_test))


mse,_ = model.evaluate(X_train, Y_train, batch_size=15)
print("mse : ", mse)
# print("loss >> ", loss)
# print("acc >> ", acc)


# 3. 예측
y_predict = model.predict(X_test)
print("[predict]\n", y_predict[170:179])
# print(len(y_predict))#180
