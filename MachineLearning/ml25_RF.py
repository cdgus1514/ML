import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


## 데이터 로드
wine = pd.read_csv("/content/winequality-white.csv", sep=";", encoding="utf-8")


## 훈련 / 시험데이터셋 분리
y = wine["quality"]
x = wine.drop("quality", axis=1)

## y 레이블 변경
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


## 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold

tree = RandomForestClassifier()


## 파라라미터
parameters = {
    "n_estimators": [10,30,100],
    "max_depth": [1,3,8],
    "bootstrap": [True, False],
    "warm_start": [True, False],
    "min_samples_split": [5, 10, 20],
    "n_jobs": [-1]
}

k_cv = KFold(n_splits=5, shuffle=True)
search = RandomizedSearchCV(tree, parameters, cv=k_cv)
search.fit(x_train, y_train)
print("최적 매개변수 >> ", search.best_estimator_)


## 평가
y_pred = search.predict(x_test)
print("최종 정답률 >> ", accuracy_score(y_test, y_pred))
last_score = search.score(x_test, y_test)
print("최종 정답률 >> ", last_score)


##############################################################################################
# 최적 매개변수 >>  RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#                        max_depth=8, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=10,
#                        min_weight_fraction_leaf=0.0, n_estimators=100,
#                        n_jobs=-1, oob_score=False, random_state=None, verbose=0,
#                        warm_start=True)
# 최종 정답률 >>  0.9418367346938775
# 최종 정답률 >>  0.9418367346938775
##############################################################################################