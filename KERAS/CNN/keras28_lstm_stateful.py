import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


## 1. 데이터 불러오기
# df = pd.read_csv("/content/kospi200test.csv", encoding="utf-8")
# df = pd.read_csv(gdp_path, sep='\t', engine='python')
df = pd.read_csv("/content/kospi200test.csv", sep=",", engine="python")

print("## 데이터 확인 ##")
print(df.head(5))
print(df.shape)
# df = df.groupby(['Date'], as_index=False)
# print(df.head(5))

print("================================================")

# kospi_df = np.array(df)
# # print(kospi_df)
# print(type(kospi_df))
# print(kospi_df.shape)       # (599, 7)
col_count = (len(kospi_df)) # (599)
# print(kospi_df[0][1:4])


# # 시가, 최고가, 최저가 분리
# train=[]
# for i in range(col_count):
#   train.append(kospi_df[i][1:4])
  

# print(train)



# ## 종가 데이터셋 분리
# close = []
# for i in range(col_count):
#   close.append(kospi_df[i][4])


# print(close)


dataset_temp = df.drop(['Date', 'dall'], axis=1)
dataset_temp = dataset_temp.as_matrix()
print(dataset_temp[col_count-1])



new_dataset = []
for i in range(col_count-1, -1, -1):
  new_dataset.append(dataset_temp[i])
  
  

  
  
# 내림차순 정렬  
new_dataset_temp = np.array(new_dataset)
print("new_dataset_temp shape : ", new_dataset_temp.shape)


## 종가 데이터셋 분리
close = []
for i in range(col_count):
  close.append(new_dataset_temp[i][3])

close = np.array(close)
# print("close shape : ", close.shape)
close = np.reshape(close, (599,1))
# print("close reshape : ", close.shape)


## 학습데이터 전처리
scaler = MinMaxScaler()
scaler.fit(new_dataset_temp)
dataset_sc = scaler.transform(new_dataset_temp)








##############################
X_train = dataset_sc[0:394]
Y_train = close[6:400]

X_test = dataset_sc[395:583]
Y_test = close[401:600]

X_train = np.reshape(X_train, (394, 5, 1))
# X_test = np.reshape(X_test, (198, 5, 1))

print(len(X_train)) #394
print(len(Y_train)) #394
print(len(X_test))  #198
print(len(Y_test))  #198




model = Sequential()
model.add(LSTM(128, batch_input_shape=(1,5,1), stateful=True))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dropout(0.5))

model.add(Dense(1))



# model.summary()
model.compile(loss="mse", optimizer="adam", metrics=["mse"])


## EarlyStopping 적용
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

## fit 구현 >> 실행 후 섞인 데이터를 그대로 유지한채 50번 반복실행
num_epochs = 50
for epoch_idx in range(num_epochs):
    print("epoch : " +str(epoch_idx))
    model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2, shuffle=False, validation_data=(X_test, Y_test), callbacks=[early_stopping_callback])
    model.reset_states()



mse, _ = model.evaluate(X_train, Y_train, batch_size=1)
print("mse : ", mse)
model.reset_states()



# 3. 예측
y_predict = model.predict(X_test, batch_size=1)
print(y_predict[0:10])   # 105, 106, 107, 108, 109

