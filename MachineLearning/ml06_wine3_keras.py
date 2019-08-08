import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping


# 1. 데이터 로드
wine = pd.read_csv("/content/winequality-white.csv", sep=";", encoding="utf-8")


## 훈련 / 시험데이터셋 분리
y = wine["quality"]
x = wine.drop("quality", axis=1)

## y레이블 변경
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

## 데이터 전처리
sc = StandardScaler()
sc.fit_transform(x_train)
sc.transform(x_test)



# 2. 모델구성
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(11,)))
model.add(Dense(128, activation="relu"))
# model.add(BatchNormalization())
model.add(Dropout(0.2))
# model.add(Dense(128, activation="relu"))
# model.add(Dense(128, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(256, activation="relu"))
# model.add(Dense(256, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
stop = EarlyStopping(monitor="loss", patience=5, mode="auto")

## one hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("x_train shape >> ", x_train.shape)   # (3918,11)
print("y_train shape >> ", y_train.shape)   # (3918,10)
print("x_test shape >> ", x_test.shape)   # (980,11)
print("y_test shape >> ", y_test.shape)   # (980,10)



# 3. 모델실행
# model.fit(x_train, y_train, epochs=100, batch_size=10)
hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), batch_size=10, callbacks=[stop])



# 4. 모델평가
loss, acc = model.evaluate(x_test, y_test)
print("loss >> ", loss)
print("acc >> ", acc)



# # 품질 데이터로 나누고 count
# count_data = wine.groupby("quality")["quality"].count()
# print(count_data)


# # 그래프
# count_data.plot()
# plt.savefig("wine-count-plt.png")
# plt.show()


# 튜닝 전
# loss >>  0.24681154513845638
# acc >>  0.9377551017975321