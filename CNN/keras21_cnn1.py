from keras.models import Sequential

filter_size = 7     # 컷팅한 이미지 개수(output)
kernel_size = (2,2) # 이미지 컷팅 사이즈

from keras.layers import Conv2D, MaxPooling2D
model = Sequential()

# 28x28=크기, 1=흑백 형태로 데이터 삽입
# default padding = valid, same= 컷팅 후 이미지 데이터 유실 방지
model.add(Conv2D(7, (2,2), padding="same", input_shape=(5,5,1)))  
model.add(Conv2D(16,(2,2)))
model.add(MaxPooling2D(2,2))    # 몇개씩 자를껀지 (10,10) >> (3,3) > (3,3) , 1 drop
model.add(Conv2D(8,(2,2)))

# model.add(Flatten())
# model.add(Dense(x))

# model.summary()