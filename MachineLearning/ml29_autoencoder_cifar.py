from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras import regularizers
import matplotlib.pyplot as plt

# minmax 정규화, standard표준화
# tensorboard


# 3채널 구성된 32x32 이미지 6만장
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_CLOS = 32



# 상수 정의
BATCH_SIZE = 200
NB_EPOCH = 1000
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2



# 데이터셋 불러오기
(X_train, _), (X_test, _) = cifar10.load_data()
print("X_train shape >> ", X_train.shape)   # (50000, 32, 32, 3)
print("X_test shape >>", X_test.shape)      # (10000, 32, 32, 3)

# # 범주형으로 변환
# Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
# Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

# 실수형으로 변환 및 정규화
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train.reshape(50000, 3072)
X_test = X_test.reshape(10000, 3072)
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = X_train.reshape(50000, 32, 32, 3)
X_test = X_test.reshape(10000, 32, 32, 3)

print("X_train shape >> ", X_train.shape)   # (50000, 3072)
print("X_test shape >>", X_test.shape)      # (10000, 3072)



# 오토인코더 모델 구성
encoding_dim = 32
drop = 0.2

# input_img = Input(shape=(3072,))
input_img = Input(shape=(32,32,3))

# encoded = Dense(encoding_dim, activation="relu")(input_img)
encoded = Conv2D(32, (3,3), input_shape=(32,32,3), activation="relu", padding="same")(input_img)

# hidden = Dense(128, activation="relu")(encoded)
# Dropout(drop)(hidden)
# hidden = Dense(128, activation="relu")(hidden)
# hidden = Dense(128, activation="relu")(hidden)
# Dropout(drop)(hidden)
# hidden = Dense(64, activation="relu")(hidden)
# hidden = Dense(64, activation="relu")(hidden)
# hidden = Dense(64, activation="relu")(hidden)
# Dropout(drop)(hidden)
# hidden = Dense(32, activation="relu")(hidden)
# hidden = Dense(32, activation="relu")(hidden)

hidden = Conv2D(32, (3,3), padding="same", activation="relu")(encoded)
hidden = MaxPooling2D((2,2), padding='same')(hidden)
hidden = BatchNormalization()(hidden)
hidden = Dropout(drop)(hidden)
hidden = Conv2D(128, (3,3), padding="same", activation="relu")(hidden)
hidden = MaxPooling2D((2,2), padding="same")(hidden)
hidden = BatchNormalization()(hidden)
hidden = Dropout(drop)(hidden)
hidden2 = Conv2D(64, (3,3), padding="same", activation="relu")(hidden)
hidden2 = UpSampling2D((2,2))(hidden2)
hidden = BatchNormalization()(hidden2)
hidden2 = Dropout(drop)(hidden2)
hidden2 = Conv2D(3, (3,3), activation="relu", padding="same")(hidden2)
hidden2 = UpSampling2D((2,2))(hidden2)
hidden = BatchNormalization()(hidden2)
hidden2 = Dropout(drop)(hidden2)

# decoded = Dense(3072, activation="sigmoid")(hidden)
decoded = Conv2D(3, (3,3), activation="sigmoid", padding="same")(hidden2)

autoencoder = Model(input_img, decoded)

autoencoder.summary()



encoder = Model(input_img, encoded)

# encoded_input = Input(shape=(32,32,3, ))
# decoded_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoded_layer(encoded_input))


autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

stop = EarlyStopping(monitor="val_loss", patience=5)

history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test), callbacks=[stop])



encoded_imgs = autoencoder.predict(X_test)
decoded_imgs = autoencoder.predict(encoded_imgs)


############ 이미지 출력 ############
import matplotlib.pyplot as plt

n = 10

plt.figure(figsize=(20,4))
for i in range(n):
    # 원본
    ax = plt.subplot(2,n,i+1)
    plt.imshow(X_test[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  
    
    # 축소 후
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(32,32,3))
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
    


plot_acc(history, "(a) 학습 경과에 따른 정확도 변화 추이")
plt.show()

plot_loss(history, "(b) 학습 경과에 따른 손실값 변화 추이")
plt.show()


loss, acc = autoencoder.evaluate(X_test, X_test)
print(loss, acc)

# 0.5950490944862366 0.012014225304685534