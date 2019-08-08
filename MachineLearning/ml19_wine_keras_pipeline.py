import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# 1. 데이터 로드
wine = pd.read_csv("/content/winequality-white.csv", sep=";", encoding="utf-8")


## 훈련 / 시험데이터셋 분리
y = wine["quality"]
x = wine.drop("quality", axis=1)

## y레이블 변경
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

# ## 데이터 전처리
# sc = StandardScaler()
# sc.fit_transform(x_train)
# sc.transform(x_test)



# 2. 모델구성
def bulid_model(optimizer="adam", drop=0.2):
    model = Sequential()
    model.add(Dense(256, activation="relu", input_shape=(11,)))
    model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Dense(16))
    model.add(Dense(16))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Dense(32))
    model.add(Dense(64))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# stop = EarlyStopping(monitor="loss", patience=5, mode="auto")

## one hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("x_train shape >> ", x_train.shape)   # (3918,11)
print("y_train shape >> ", y_train.shape)   # (3918,10)
print("x_test shape >> ", x_test.shape)   # (980,11)
print("y_test shape >> ", y_test.shape)   # (980,10)



## 튜닝
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


kfold_cv = KFold(n_splits=5, shuffle=True)
model = KerasClassifier(build_fn=bulid_model)

prameter = {
    "model__batch_size": [5,15,55],
    "model__optimizer": ["adam", "adadelta", "rmsprop"],
    "model__keep_prob": [0,0.2,0.5],
    "model__epochs": [50, 100]
}

pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

search = RandomizedSearchCV(pipe, prameter, cv=kfold_cv)
search.fit(x_train, y_train)

print(search.best_params_)

# from sklearn.metrics import accuracy_score
# y_pred = search.predict(x_test)
# print('정답률 >> ', accuracy_score(y_test, y_pred))

