import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# 1. 데이터
df = pd.read_csv("/content/tem10y.csv", encoding="utf-8")


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



## 모델
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import warnings

tree = GradientBoostingRegressor()


## 파라라미터
parameters = {
    "n_estimators": [10,30,100,150,300],
    "max_depth": [1,3,8],
    "warm_start": [True, False],
    "min_samples_split": [5, 10, 20],
}

k_cv = KFold(n_splits=5, shuffle=True)
search = RandomizedSearchCV(tree, parameters, cv=k_cv)

warnings.filterwarnings("ignore")
search.fit(x_train, y_train)
print("훈련 세트 정확도 >> {:.3f}".format(search.score(x_train, y_train)))  # 
print("테스트 세트 정확도 >> {:.3f}".format(search.score(x_test, y_test)))  # 
print("최적 매개변수 >> ", search.best_estimator_)


## 평가
y_pred = search.predict(x_test)
last_score = search.score(x_test, y_test)
print("\n최종 정답률 >> ", last_score)


##############################################################################################
# 최적 매개변수 >>  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#                           learning_rate=0.1, loss='ls', max_depth=3,
#                           max_features=None, max_leaf_nodes=None,
#                           min_impurity_decrease=0.0, min_impurity_split=None,
#                           min_samples_leaf=1, min_samples_split=5,
#                           min_weight_fraction_leaf=0.0, n_estimators=150,
#                           n_iter_no_change=None, presort='auto',
#                           random_state=None, subsample=1.0, tol=0.0001,
#                           validation_fraction=0.1, verbose=0, warm_start=True)
# 최종 정답률 >>  0.9198979616856161
##############################################################################################