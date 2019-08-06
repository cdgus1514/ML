import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score


# 데이터 로드
# dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
iris_data = pd.read_csv("/content/iris.csv", names=["SepalLenght", "SepalWidth", "PetalLenght", "petalWidth", "Name"], encoding="utf-8")
# print(iris_data)
print("iris shape >> ", iris_data.shape)

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLenght", "SepalWidth", "PetalLenght", "petalWidth"]]
print("x shape >> ", x.shape)
print("y shape >> ", y.shape)



# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8, shuffle=True)
# print(len(x_train))
# print(len(y_train))
# print(len(x_test))
# print(len(x_test))
# print(y_test)



# 모델구성 및 실행
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)
model = LinearSVC()
model.fit(x_train, y_train)



# 평가
y_pred = model.predict(x_test)
print("\n정답률 >> ", accuracy_score(y_test, y_pred))
# print(y_pred)

