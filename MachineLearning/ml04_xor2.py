### 머신러닝 XOR
### Knn을 사용한 분류모델

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터셋 생성
learn_data = [[0,0], [1,0], [0,1], [1,1]]
learn_label = [0,1,1,0]


# 2. 모델생성
# model = svm.SVC()
model = KNeighborsClassifier(n_neighbors=1)


# 3. 실행
model.fit(learn_data, learn_label)


# 4. 평가, 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)


print(x_test, "예측결과 >> ", y_predict)
# 0110과 y_predict 결과값 비교해서 정확도 측정
print("acc >> ", accuracy_score([0,1,1,0], y_predict))

