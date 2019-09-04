import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

xy = np.load("mnist_train.npz")
x_train = xy["x_train"]
y_train = xy["y_train"]

xy = np.load("mnist_test.npz")
x_test = xy["x_test"]
y_test = xy["y_test"]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# x_train = np.reshape(x_train, (60000,784,1))
# x_test = np.reshape(x_test, (10000,784,1))
# print(x_train.shape, y_train.shape) # (120,4) (120,3)
# print(x_test.shape, y_test.shape)   # (30,4) (30,3)


model = Sequential()
model.add(LSTM(32, input_shape=(28,28), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# 실행
model.fit(x_train, y_train, epochs=100, batch_size=256)



# 평가
loss, acc = model.evaluate(x_test, y_test)
print("test_acc", acc)