from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. 데이터셋 생성
learn_data = [[0,0], [1,0], [0,1], [1,1]]
learn_label = [0,0,0,1]
learn_data = np.array(learn_data)
learn_label = np.array(learn_label)
# print(learn_data.shape)   # (4,2)
# print(learn_label.shape)  # (4,)



# 2. 모델생성
model = Sequential()
# model.add(Dense(10, input_shape=(2,), activation="relu"))
model.add(Dense(10, input_dim=2, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(15, activation="relu"))
# model.add(Dense(1, kernel_initializer="normal", activation="softmax"))
# model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))



# 3. 실행
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(learn_data, learn_label, epochs=200)



# 4. 평가
loss, acc = model.evaluate(learn_data, learn_label, batch_size=1)
print("loss >> ", loss)
print("acc >> ", acc)


x_test = [[0,0], [1,0], [0,1], [1,1]]
x_test = np.array(x_test)
y_predict = model.predict(learn_data)

print(x_test.shape)     # (4,2)
print(y_predict.shape)  # (4,1)

print("예측결과 \n", y_predict)
