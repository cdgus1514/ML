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



# 2. 모델
from keras.layers import Input, Dense, Dropout
from keras.models import Model

# 인코딩 데이터 크기 설정
encoding_dim = 32
drop = 0.5

# 입력 플레이스홀더
input_img = Input(shape=(784,))             # Input 784
# 인코딩된 표현 (히든레이어 input=32)
encoded = Dense(encoding_dim, activation="relu")(input_img)

hidden = Dense(128, activation="relu")(encoded)
Dropout(drop)(hidden)
hidden = Dense(128, activation="relu")(hidden)
hidden = Dense(128, activation="relu")(hidden)
Dropout(drop)(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
Dropout(drop)(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)

# 입력의 손실이 있는 재구성
decoded = Dense(784, activation="sigmoid")(encoded)
# decoded = Dense(784, activation="relu")(encoded)

# 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)     # 784 >> 32 >> 784
# 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded)         # 784 >> 32


# 인코딩된 입력을 입력 (히든레이어의 ~를 인풋으로 사용)
encoded_input = Input(shape=(encoding_dim,))
# 오토 인코더 모델의 마지막 레이어
decoded_layer = autoencoder.layers[-1]
# 디코더 모델 생성
decoder = Model(encoded_input, decoded_layer(encoded_input))        # Output 32 >> 784



autoencoder.summary()
encoder.summary()       # 784 >> 32
decoder.summary()       # 32 >> 784


autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# autoencoder.compile(optimizer="adadelta", loss="mse", metrics=["accuracy"])

# 같은 값의 x, y를 모델에 넣음 >> 다른 값 출력
history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))


#
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs)
print(decoded_imgs)
print(encoded_imgs.shape)   # (10000,32)
print(decoded_imgs.shape)   # (10000,784)


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