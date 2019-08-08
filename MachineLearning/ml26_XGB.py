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
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

tree = XGBRegressor()

## 파라라미터
parameters = {
    "booster": ["gbtree"],
    "n_estimators": [30,100,350,1000],
    "min_child_weight": [1,5,10],
    "max_depth": [1,3,8],
    "n_jobs": [-1],
    "gamma": [0,1,5,10],
    "alpha": [1,5,10],
    "objective":["reg:squarederror"]
}

k_cv = KFold(n_splits=5, shuffle=True)
search = RandomizedSearchCV(tree, parameters, cv=k_cv)
search.fit(x_train, y_train)
print("훈련 세트 정확도 >> {:.3f}".format(search.score(x_train, y_train)))  # 1.00
print("테스트 세트 정확도 >> {:.3f}".format(search.score(x_test, y_test)))  # 0.937
print("최적 매개변수 >> ", search.best_estimator_)


## 평가
y_pred = search.predict(x_test)
last_score = search.score(x_test, y_test)
print("\n최종 정답률 >> ", last_score)


##############################################################################################
# 최적 매개변수 >>  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=10,
#              importance_type='gain', learning_rate=0.1, max_delta_step=0,
#              max_depth=3, min_child_weight=5, missing=None, n_estimators=300,
#              n_jobs=-1, nthread=None, objective='reg:squarederror',
#              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#              seed=None, silent=None, subsample=1, verbosity=1)
# 최종 정답률 >>  0.9194724292739169
##############################################################################################