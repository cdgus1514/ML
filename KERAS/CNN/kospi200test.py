import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


## 1. 데이터 불러오기 >> csv파일 업로드 필요
# df = pd.read_csv("/content/kospi200test.csv", encoding="utf-8")
df = pd.read_csv("/content/kospi200test.csv", sep=",", engine="python")

print("## 데이터 확인 ##")
print(df.head(5))
print("original shape : ", df.shape)  #(600,7)

print("================================================")

# dataset_temp = df.drop(['Date', 'dall'], axis=1)
dataset_temp = df.drop(['Date', 'Volume', 'dall'], axis=1)
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
  close.append(new_dataset_temp[i][3])

close = np.array(close)
print("close shape : ", close.shape)
close = np.reshape(close, (col_count,1))
print("close reshape : ", close.shape)

print(new_dataset_temp)



## 데이터셋 스케일링
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(new_dataset_temp)
dataset_sc = scaler.transform(new_dataset_temp)


# 학습, 시험데이터셋 분할
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(dataset_sc, close, random_state=66, test_size=0.3)   # traint : test >> 7:3
# X_train = dataset_sc[0:394]
# Y_train = close[6:400]

# X_test = dataset_sc[395:593]
# Y_test = close[401:600]

# X_train = np.reshape(X_train, (394, 5, 1))
# X_test = np.reshape(X_test, (198, 5, 1))


print("X_train shape >> ", X_train.shape, len(X_train))   # (420,5)
print("Y_train shape >> ", Y_train.shape, len(Y_train))   # (420,1)
print("X_test shape >> ", X_test.shape, len(X_test))      # (180,5)
print("Y_test shape >> ", Y_test.shape, len(Y_test))      # (180,1)





model = Sequential()
# model.add(LSTM(128, batch_input_shape=(1,5,1), stateful=True, return_sequences=True))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(LSTM(128, batch_input_shape=(1,5,1), stateful=True))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(1024))
# model.add(Dropout(0.5))
# model.add(Dense(1024))
# model.add(Dropout(0.5))
# model.add(Dense(1024))
# model.add(Dropout(0.5))



# model.add(Dense(30, activation="relu", input_shape=(5,)))
model.add(Dense(30, activation="relu", input_shape=(4,)))

for i in range(3):
  model.add(Dense(30))
  model.add(Dense(45))
  model.add(Dense(70))
  model.add(Dense(30))
  model.add(Dense(10))


model.add(Dense(1))



# model.summary()
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])


## EarlyStopping 적용
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="auto")



model.fit(X_train, Y_train, epochs=2000, batch_size=15, verbose=1, validation_data=(X_test, Y_test), callbacks=[early_stopping_callback])



loss, acc = model.evaluate(X_train, Y_train, batch_size=1)
# print("mse : ", mse)
print("loss >> ", loss)
print("acc >> ", acc)




# 3. 예측
y_predict = model.predict(X_test)
# print(len(y_predict))#180
print("predict >> ", y_predict[179])
# print("predict >> ", y_predict[119])



from sklearn.metrics import r2_score

r2_y_predict = r2_score(Y_test, y_predict)
print("R2 >> ", r2_y_predict)