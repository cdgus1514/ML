### 머신러닝 AND모델 >> 분류
### SVM 서포트 백터 머신 >> 분류용 선형모델

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터셋 생성
learn_data = [[0,0], [1,0], [0,1], [1,1]]
learn_label = [0,0,0,1]


# 2. 모델생성
# clf = LinearSVC()
model = LinearSVC()


# 3. 실행
# clf.fit(learn_data, learn_label)
model.fit(learn_data, learn_label)


# 4. 평가, 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
# y_predict = clf.predict(x_test)
y_predict = model.predict(x_test)


print("예측결과 >> ", y_predict)
print(x_test.shape)
print(y_predict.shape)
# 0001과 y_predict 결과값 단순비교 >> 정확도 측정
print("acc >> ", accuracy_score([0,0,0,1], y_predict))  

