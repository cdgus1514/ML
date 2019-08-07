import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# 1. 데이터 로드
# iris_data = pd.read_csv("C:/Users/CDH/Downloads/Data/iris2.csv", encoding="utf-8")
iris_data = pd.read_csv("/content/iris2.csv", encoding="utf-8")


## 레이블, 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


## 훈련, 시험데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


## 그리드서치 매개변수
parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]
## 랜덤포레스트 매개변수
parameters2 = [
    {"n_estimators":[10, 100, 1000], "random_state":[10,100,1000], "oob_score":[True]},
    {"n_estimators":[10, 100, 1000], "random_state":[10,100,1000], "oob_score":[False],"bootstrap":[False]}
]
## 케이네이버 매개변수
parameters3 = [
    {"n_neighbors":[1,10,55], "leaf_size":[15,45,65], "p":[5,10,25]}
]


## 그리드서치
kfold_cv = KFold(n_splits=5, shuffle=True)
# clf = GridSearchCV(SVC(), parameters, cv=kfold_cv)
# clf.fit(x_train, y_train)
# print("최적의 매개변수 >> ", clf.best_estimator_)

## 케이네이버
clf = GridSearchCV(KNeighborsClassifier(), parameters3, cv=kfold_cv)
clf.fit(x_train, y_train)
print("최적의 매개변수 >> ", clf.best_estimator_)



## 최적의 매개변수로 평가
y_pred = clf.predict(x_test)
print("최종 정답률 >> ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 >> ", last_score)

'''
최적의 매개변수 >>  KNeighborsClassifier(algorithm='auto', leaf_size=15, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=10, p=25,
                     weights='uniform')
최종 정답률 >>  1.0
최종 정답률 >>  1.0
'''