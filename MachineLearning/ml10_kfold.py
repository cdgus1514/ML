import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# 1. 데이터 로드
iris_data = pd.read_csv("C:/Users/CDH/Downloads/Data/iris2.csv", encoding="utf-8")
# iris_data = pd.read_csv("/content/iris2.csv", encoding="utf-8")


## 레이블, 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


## classifier 알고리즘 모두 추출
warnings.filterwarnings("ignore")
allAlgorithms = all_estimators(type_filter="classifier")


## K-분할 크로스 발리데이션 전용 객체
kfold_cv = KFold(n_splits=10, shuffle=True)  # 3-10
num = []

for(name, algorithm) in allAlgorithms:
    #각 알고리즘 객체 생성
    clf = algorithm()

    ## score 매서드를 가진 클래스를 대상을 하기
    if hasattr(clf, "score"):

        # 크로스 발리데이션
        scores = cross_val_score(clf, x, y, cv=kfold_cv)
        print(name,"의 정답률 >> ")
        # print(scores)
        
        s = np.array(scores)
        avg = np.mean(s)
        num.append(s)
        print(avg)



'''
## Top score ##

n_splits = 3 >> SVC, QuadraticDiscriminantAnalysi, MLPClassifier
n_splist = 4 >> SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
n_splist = 5 >> SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysi
n_splist = 6 >> SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
n_splist = 7 >> SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
n_splist = 8 >> SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
n_splist = 9 >> SVC, QuadraticDiscriminantAnalysis, MLPClassifier
n_splist = 10 >> SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

'''