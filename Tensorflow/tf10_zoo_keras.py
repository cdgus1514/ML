import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# data load
xy = np.loadtxt("C:/Study/ML/Data/data-04-zoo.csv", delimiter=",", dtype=np.float32)
x = xy[:, 0:-1]
y = xy[:,[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


## one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)    # (80,16)
print(y_train.shape)    # (80,7)
print(x_test.shape)     # (21,16)
print(y_test.shape)     # (21,7)


# model
model = Sequential()

model.add(Dense(7, input_dim=16, activation="softmax"))
# model.add(Dense(7, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test), batch_size=14)


# evaluate
loss, acc = model.evaluate(x_test, y_test)
print("loss >>", loss)
print("acc >>", acc)



'''
loss >> 0.5264562368392944
acc >> 0.8571428656578064
'''
