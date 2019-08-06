from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 로드
# dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
dataset = numpy.loadtxt("/content/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:7]
Y = dataset[:, 8]

indices = numpy.arange(X.shape[0])
numpy.random.shuffle(indices)
X = X[indices]
y = y[indices]
X_train = X[:700]
y_train = y[:700]
X_test = X[700:]
y_test = y[700:]


# 모델생성
model = KNeighborsClassifier(n_neighbors=1)


#실행
model.fit(X_train, y_train)


# 평가, 예측
y_predict = model.predict(X_test)
print("acc >> " % (accuracy_score(X_test, Y)))