from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])


## 직선 회귀 분석
lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y)            #학습
pre_y = lr.predict(test_x)          #예측

score = lr.score(test_x, pre_y)
print("score >> ", score)

# 결과를 그래프로 그리기
plt.figure(figsize=(10,6), dpi=100)
plt.plot(test_y, c="r")
plt.plot(pre_y, c="r")
plt.savefig("tenki-kion-lr.png")
plt.show()