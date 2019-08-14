import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class iris_npz:
    def create_npz():
        iris_data = pd.read_csv("/content/iris.csv", names=["SepalLenght", "SepalWidth", "PetalLenght", "petalWidth", "Name"], encoding="utf-8")
    #     iris_data = pd.read_csv("C:/CDH/AI/ML/Data/iris.csv", names=["SepalLenght", "SepalWidth", "PetalLenght", "petalWidth", "Name"], encoding="utf-8")

        from sklearn.preprocessing import LabelEncoder
        x = iris_data.iloc[:, 0:4].values
        y = iris_data.iloc[:, 4].values

        encoder = LabelEncoder()
        y1 = encoder.fit_transform(y)
        y = pd.get_dummies(y1).values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)


        ### iris datasets save ###
        np.savez("iris_train.npz", x_train=x_train, y_train=y_train)
        np.savez("iris_test.npz", x_test=x_test, y_test=y_test)
        print("! iris_npz 파일 생성")

        print("--------------------------------------------------")

        ## iris datasets load ###
        iris_load = np.load("iris_train.npz")
        X_train = iris_load["x_train"]
        Y_train = iris_load["y_train"]

        iris_load = np.load("iris_test.npz")
        X_test = iris_load["x_test"]
        Y_test = iris_load["y_test"]

        print("X_train shape >> ", X_train.shape)   # (120,4)
        print("Y_train shape >> ", Y_train.shape)   # (120,3)
        print("X_test shape >> ", X_test.shape)     # (30,4)
        print("Y_test shape >> ", Y_test.shape)     # (30,3)



class mnist_npz:
    def create_npz():
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)


        ## mnist datasets save ###
        np.savez("mnist_train.npz", x_train=x_train, y_train=y_train)
        np.savez("mnist_test.npz", x_test=x_test, y_test=y_test)
        print("! mnist_npz 파일 생성")

        print("--------------------------------------------------")

        ### mnist datases load ###
        mnist_load = np.load("mnist_train.npz")
        X_train = mnist_load["x_train"]
        Y_train = mnist_load["y_train"]

        mnist_load = np.load("mnist_test.npz")
        X_test = mnist_load["x_test"]
        Y_test = mnist_load["y_test"]


        print("X_train shape >> ", X_train.shape)   # (60000,28,28)
        print("Y_train shape >> ", Y_train.shape)   # (60000,)
        print("X_test shape >> ", X_test.shape)     # (10000,28,28)
        print("Y_test shape >> ", Y_test.shape)     # (10000,)



class cifar10_npz:
    def create_npz():
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)


        ## cifar10 datasets save ###
        np.savez("cifar10_train.npz", x_train=x_train, y_train=y_train)
        np.savez("cifar10_test.npz", x_test=x_test, y_test=y_test)
        print("! cifar10_npz 파일 생성")

        print("--------------------------------------------------")

        ### cifar10 datases load ###
        cifar10_load = np.load("cifar10_train.npz")
        X_train = cifar10_load["x_train"]
        Y_train = cifar10_load["y_train"]

        cifar10_load = np.load("cifar10_test.npz")
        X_test = cifar10_load["x_test"]
        Y_test = cifar10_load["y_test"]


        print("X_train shape >> ", X_train.shape)   # (50000,32,32,3)
        print("Y_train shape >> ", Y_train.shape)   # (50000,1)
        print("X_test shape >> ", X_test.shape)     # (10000,32,32,3)
        print("Y_test shape >> ", Y_test.shape)     # (10000,1)



class boston_npz:
    def create_npz():
        from keras.datasets import boston_housing
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)


        ## mnist datasets save ###
        np.savez("boston_housing_train.npz", x_train=x_train, y_train=y_train)
        np.savez("boston_housing_test.npz", x_test=x_test, y_test=y_test)
        print("! boston_npz 파일 생성")

        print("--------------------------------------------------")

        ### mnist datases load ###
        boston_housing_load = np.load("boston_housing_train.npz")
        X_train = boston_housing_load["x_train"]
        Y_train = boston_housing_load["y_train"]

        boston_housing_load = np.load("boston_housing_test.npz")
        X_test = boston_housing_load["x_test"]
        Y_test = boston_housing_load["y_test"]


        print("X_train shape >> ", X_train.shape)   # (404,13)
        print("Y_train shape >> ", Y_train.shape)   # (404,)
        print("X_test shape >> ", X_test.shape)     # (102, 13)
        print("Y_test shape >> ", Y_test.shape)     # (102,)



class wine_npz:
    def create_npz():
        # wine_data = pd.read_csv("/content/winequality-white.csv", sep=";", encoding="utf-8")
        wine_data = pd.read_csv("C:/CDH/AI/ML/Data/winequality-white.csv", sep=";", encoding="utf-8")

        ## 훈련 / 시험데이터셋 분리
        y = wine_data["quality"]
        x = wine_data.drop("quality", axis=1)

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
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)


        ## mnist datasets save ###
        np.savez("wine_train.npz", x_train=x_train, y_train=y_train)
        np.savez("wine_test.npz", x_test=x_test, y_test=y_test)
        print("! wine_npz 파일 생성")

        print("--------------------------------------------------")

        ### mnist datases load ###
        wine_load = np.load("wine_train.npz")
        X_train = wine_load["x_train"]
        Y_train = wine_load["y_train"]

        wine_load = np.load("wine_test.npz")
        X_test = wine_load["x_test"]
        Y_test = wine_load["y_test"]


        print("X_train shape >> ", X_train.shape)   # (3918,11)
        print("Y_train shape >> ", Y_train.shape)   # (3918,)
        print("X_test shape >> ", X_test.shape)     # (980,11)
        print("Y_test shape >> ", Y_test.shape)     # (980,)



class cancer_npz:
    def create_npz():
        from sklearn.datasets import load_breast_cancer

        cancer = load_breast_cancer()
        x = cancer.data
        y = cancer.target

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8, shuffle=True)
        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)

        ## mnist datasets save ###
        np.savez("cancer_train.npz", x_train=x_train, y_train=y_train)
        np.savez("cancer_test.npz", x_test=x_test, y_test=y_test)
        print("! caner_npz 파일 생성")

        print("--------------------------------------------------")

        ### mnist datases load ###
        cancer_load = np.load("cancer_train.npz")
        X_train = cancer_load["x_train"]
        Y_train = cancer_load["y_train"]

        cancer_load = np.load("cancer_test.npz")
        X_test = cancer_load["x_test"]
        Y_test = cancer_load["y_test"]


        print("X_train shape >> ", X_train.shape)   # (455,30)
        print("Y_train shape >> ", Y_train.shape)   # (455,)
        print("X_test shape >> ", X_test.shape)     # (114,30)
        print("Y_test shape >> ", Y_test.shape)     # (114,)



class weather_npz:
    def create_npz():
        # df = pd.read_csv("/content/tem10y.csv", encoding="utf-8")
        df = pd.read_csv("C:/CDH/AI/ML/Data/tem10y.csv", encoding="utf-8")

        ## 데이터 분할
        train_year = (df["연"] <= 2015)
        test_year = (df["연"] >= 2016)
        interval = 6


        ## 과거 6일의 데이터를 기반으로 학습할 데이터 생성
        def make_data(data):
            x = []  #학습데이터셋
            y = []  #결과
            temps = list(data["기온"])

            for i in range(len(temps)):
                y.append(temps[i])
                xa = []

                for p in range(interval):
                    d = i+p-interval
                    xa.append(temps[d])
                
                x.append(xa)
            
            return (x, y)

        x_train, y_train = make_data(df[train_year])
        x_test, y_test = make_data(df[test_year])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print("x_train shape >> ", x_train.shape)
        print("y_train shape >> ", y_train.shape)
        print("x_test shape >> ", x_test.shape)
        print("y_test shape >> ", y_test.shape)

        ## mnist datasets save ###
        np.savez("weather_train.npz", x_train=x_train, y_train=y_train)
        np.savez("weather_test.npz", x_test=x_test, y_test=y_test)
        print("! weather_npz 파일 생성")

        print("--------------------------------------------------")

        ### mnist datases load ###
        weather_load = np.load("weather_train.npz")
        X_train = weather_load["x_train"]
        Y_train = weather_load["y_train"]

        weather_load = np.load("weather_test.npz")
        X_test = weather_load["x_test"]
        Y_test = weather_load["y_test"]


        print("X_train shape >> ", X_train.shape)   # (3652,6)
        print("Y_train shape >> ", Y_train.shape)   # (3652,)
        print("X_test shape >> ", X_test.shape)     # (366,6)
        print("Y_test shape >> ", Y_test.shape)     # (366,)



if __name__ == "__main__":

    create = iris_npz()
    # iris_npz.create_npz()

    create = mnist_npz()
    # mnist_npz.create_npz()

    create = cifar10_npz()
    # cifar10_npz.create_npz()
    
    create = boston_npz()
    # boston_npz.create_npz()
    
    create = wine_npz()
    # wine_npz.create_npz()
    
    create = cancer_npz()
    # cancer_npz.create_npz()
    
    create = weather_npz()
    # weather_npz.create_npz()



##################################################
# !USAGE!
#
#
# 1. npz 데이터셋 불러오기                               
# 변수명 = np.load("파일명.npz")
# 
# x_train = 변수명["x_train"]
# y_test = 변수명["y_test"]
#
#
##################################################