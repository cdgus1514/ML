from keras.models import Sequential

filter_size = 32
kernel_size = (3,3) # 이미지 컷팅 사이즈

from keras.layers import Conv2D
model = Sequential()
model.add(Conv2D(filter_size, kernel_size, input_shape=(28,28,1)))  # 28x28=크기, 1=흑백
model.add(Conv2D(16,(3,3)))
model.add(Conv2D(8,(2,2)))

model.summary()
