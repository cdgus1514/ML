import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


## 데이터 로드
wine = pd.read_csv("/content/winequality-white.csv", sep=";", encoding="utf-8")


## 훈련 / 시험데이터셋 분리
y = wine["quality"]
x = wine.drop("quality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# 학습하기
## n_estimators : 모델의 독립적인 트리 개수
model = RandomForestClassifier(n_estimators=100, random_state=2)    # n_estimators=100 >> 70
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)


# 평가
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률 >> ", accuracy_score(y_test, y_pred))
print("score >> ", aaa)