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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold

tree = GradientBoostingClassifier()


## 파라라미터
parameters = {
    "n_estimators": [10,30,100,150,300],
    "max_depth": [1,3,8],
    "warm_start": [True, False],
    "min_samples_split": [5, 10, 20],
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
# 최적 매개변수 >>  GradientBoostingClassifier(criterion='friedman_mse', init=None,
#                            learning_rate=0.1, loss='deviance', max_depth=8,
#                            max_features=None, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_samples_split=5,
#                            min_weight_fraction_leaf=0.0, n_estimators=150,
#                            n_iter_no_change=None, presort='auto',
#                            random_state=None, subsample=1.0, tol=0.0001,
#                            validation_fraction=0.1, verbose=0,
#                            warm_start=False)
# 최종 정답률 >>  0.939795918367347
# 최종 정답률 >>  0.939795918367347
##############################################################################################