import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



# 1. 데이터 로드
# iris_data = pd.read_csv("C:/Users/CDH/Downloads/Data/iris2.csv", encoding="utf-8")
iris_data = pd.read_csv("/content/iris2.csv", encoding="utf-8")


## 레이블, 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


## 훈련, 시험데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(x_train, y_train)

print("테스트 점수 >> ", pipe.score(x_test, y_test))