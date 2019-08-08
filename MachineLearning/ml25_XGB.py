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
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold

tree = XGBClassifier()


## 파라라미터
parameters = {
    "booster": ["gbtree"],
    "n_estimators": [10,60,100,150,300],
    "min_child_weight": [1,5,10],
    "max_depth": [1,3,8],
    "n_jobs": [-1],
    "gamma": [0,1,5,10],
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
# 최적 매개변수 >>  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               learning_rate=0.1, max_delta_step=0, max_depth=8,
#               min_child_weight=5, missing=None, n_estimators=150, n_jobs=-1,
#               nthread=None, objective='multi:softprob', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=1, verbosity=1)
# 최종 정답률 >>  0.9418367346938775
# 최종 정답률 >>  0.9418367346938775
##############################################################################################