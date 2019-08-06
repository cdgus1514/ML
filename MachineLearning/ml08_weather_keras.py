import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
df = pd.read_csv("/content/tem10y.csv", encoding="utf-8")


## 데이터 분할
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6


## 과거 6일의 데이터를 기반으로 학습할 데이터 생성
def make_data(data):
    x = []  #학습데이터셋
    y = []  #결과
    temps = list(data["기온"])

    for i in range(len(temps)):
        y.append(temps[i])
        xa = []

        for p in range(interval):
            d = i+p-interval
            xa.append(temps[d])
        
        x.append(xa)
    
    return (x, y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])


## 데이터 전처리
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

print("train_x shape >>", train_x.shape)    # (3652,6)
print("test_x shape >>", test_x.shape)      # (366,6)
print("train_y shape >>", train_y.shape)    # (3652,)
print("test_y shape >>", test_y.shape)      # (366,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = MinMaxScaler()
# sc = StandardScaler()
sc.fit_transform(train_x)
sc.transform(test_x)

train_x_sc = np.reshape(train_x, (3652,6,1))
test_x_sc = np.reshape(test_x, (366,6,1))


# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

model = Sequential()

# model.add(LSTM(15, input_shape=(6,1), return_sequences=True))
model.add(LSTM(15, input_shape=(6,1)))
# model.add(LSTM(15))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
st = EarlyStopping(monitor="loss", patience=10, mode="auto")
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])


# 3. 실행
model.fit(train_x_sc, train_y, epochs=1000, batch_size=10, callbacks=[st])



# 4. 평가
loss, acc = model.evaluate(test_x_sc, test_y)
print("loss >> ", loss)
print("acc >> ", acc)

y_pre = model.predict(test_x_sc)
print("predict >> ", y_pre)


from sklearn.metrics import r2_score

r2_y_predict = r2_score(test_y, y_pre)
print("R2 : ", r2_y_predict)