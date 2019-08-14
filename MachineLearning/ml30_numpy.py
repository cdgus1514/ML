import numpy as np
a = np.arrange(10)
print(a)

np.save("aaa.npy", a)

b = np.load("aaa.npy")
print(b)



'''
############################## 모델 저장 ##############################
model.save("savetest01.h5")

############################## 모델 불러오기 ##############################
from keras.models import load_model
model = load_model("savetest01.h5")

## 불러온 모델에 레이어 추가
from keras.layers import Dense
model.add(Dense(1))

############################## csv 불러오기 ##############################
변수명 = numpy.loadtxt("경로/파일명.csv", delimiter=",")

변수명 = pd.read_csv("경로/파일명.csv", encoding="utf-8")

# index_col = 0, encoding="cp949", sep=",", header=None     > column 0부터 시작
# names = ["column1", "column2", ..., "Y"]



############################## 샘플 데이터셋 ##############################
from keras.datasets import mnist, cifar10, boston_housing

## keras에서 불러오면 numpy로 불러옴
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()


from sklearn.datasets import load_boston, load_breast_cancer

boston = load_boston()
print(boston.keys())
#boston.data
#boston.target

cancer = load_breast_cancer()
print(cancer.keys())
#cancer.data
#cancer.target
'''