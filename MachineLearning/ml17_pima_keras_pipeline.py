from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 로드
# dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
dataset = numpy.loadtxt("/content/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, train_size=0.8, shuffle=True)
print("x_train shape >> ", x_train.shape)   # (455,30)
print("y_train shape >> ", y_train.shape)   # (455,)
print("x_test shape >> ", x_test.shape)     # (114,30)
print("y_test shape >> ", y_test.shape)     # (114,)


# 모델설정
def bulid_model(drop=0.2, optimizer="adam"):
    model = Sequential()

    model.add(Dense(64, input_dim=8, activation="relu"))
    model.add(Dropout(drop))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(drop))
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(drop))
    
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model



# 튜닝
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

k_cv = KFold(n_splits=5, shuffle=True)
model = KerasClassifier(build_fn=bulid_model)

parameters = {
    "model__batch_size": [1,5,7,15,35],
    "model__optimizer": ["adam", "adadelta", "rmsprop"],
    "model__drop": [0,0.2,0.5],
    "model__epochs": [100, 200]
}

pipe = Pipeline([
    ("scaler", MinMaxScaler()), ("model", model)
])

search = RandomizedSearchCV(pipe, parameters, cv=k_cv)
search.fit(x_train, y_train)


print(search.best_params_)

from sklearn.metrics import accuracy_score
y_pred = search.predict(x_test)
print('정답률 >> ', accuracy_score(y_test, y_pred))

