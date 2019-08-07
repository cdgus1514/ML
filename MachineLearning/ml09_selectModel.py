import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings("ignore")


# 1. 데이터 로드
iris_data = pd.read_csv("/content/iris2.csv", encoding="utf-8")


## 레이블, 데이터 분리
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


## 훈련,시험데이터셋 분리
warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)


## classifier 알고리즘 모두 추출
warnings.filterwarnings("ignore")
allAlgorithms = all_estimators(type_filter="classifier")

for(name, algorithm) in allAlgorithms:
    #각 알고리즘 객체 생성
    clf = algorithm()

    #학습 후 평가
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name,"의 정답률 >> ", accuracy_score(y_test, y_pred))