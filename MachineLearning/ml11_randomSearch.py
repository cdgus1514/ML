import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



# 1. 데이터 로드
iris_data = pd.read_csv("C:/Users/CDH/Downloads/Data/iris2.csv", encoding="utf-8")
# iris_data = pd.read_csv("/content/iris2.csv", encoding="utf-8")


## 레이블, 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


## 훈련, 시험데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


## 그리드서치 매개변수
# parameters = [
#     {"C":[1,10,100,1000], "kernel":["linear"]},
#     {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
#     {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
# ]

def parm():
    C = [1,10,100,1000]
    kernel = ['linear', 'rbf', 'sigmoid']
    gamma = [0.001, 0.0001]
    return{"C":C, "kernel":kernel, "gamma":gamma}

parameters = parm()


## 그리드서치
kfold_cv = KFold(n_splits=5, shuffle=True)
# clf = GridSearchCV(SVC(), parameters, cv=kfold_cv)
clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print("최적의 매개변수 >> ", clf.best_estimator_)


## 최적의 매개변수로 평가
y_pred = clf.predict(x_test)
print("최종 정답률 >> ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 >> ", last_score)


'''
최적의 매개변수 >>  SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
최종 정답률 >>  0.9333333333333333
최종 정답률 >>  0.9333333333333333
'''