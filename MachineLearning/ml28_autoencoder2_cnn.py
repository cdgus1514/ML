from keras.datasets import mnist
import numpy as np

# 비지도학습 데이터 불러오기
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)    # (60000,784)
print(x_test.shape)     # (10000,784)

x_train = np.reshape(x_train, (len(x_train), 28,28,1))
x_test = np.reshape(x_test, (len(x_test), 28,28,1))




# 2. 모델
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping

# 인코딩 데이터 크기 설정
encoding_dim = 32


# 입력 플레이스홀더
input_img = Input(shape=(28,28,1))

x = Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
x = MaxPool2D((2,2), padding="same")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
x = MaxPool2D((2,2), padding="same")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
encoded = MaxPool2D((2,2), padding="same")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(encoded)


x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Conv2D(128, (3,3), activation="relu")(x)
x = UpSampling2D((2,2))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)


decoded = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

autoencoder = Model(input_img, decoded)



autoencoder.summary()

autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["accuracy"])

stop = EarlyStopping(monitor="val_loss", patience=5)

# 같은 값의 x, y를 모델에 넣음 >> 다른 값 출력
history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test), callbacks=[stop])



############ 이미지 출력 ############
import matplotlib.pyplot as plt

n = 10

plt.figure(figsize=(20,4))
for i in range(n):
    # 원본
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  
  
    # 축소 후
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



############ 그래프 ############
def plot_acc(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history["acc"])
    plt.plot(history["val_acc"])

    if title is not None:
        plt.title(title)
    
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["Traning data", "Validation data"], loc=0)
    #plt.show()


def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history["loss"])
    plt.plot(history["val_loss"])

    if title is not None:
        plt.title(title)
    
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["Traning data", "Validation data"], loc=0)
    #plt.show()


plot_acc(history, "(a) 학습 경과에 따른 정확도 변화 추이")
plt.show()

plot_loss(history, "(b) 학습 경과에 따른 손실값 변화 추이")
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)