from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# print("train_data shape : ", train_data.shape)        # 404,13
# print("train_targets shape : ", train_targets.shape)  # 404
# print("test_data shape : ", test_data.shape)          # 102,13
# print("test_targets shape : ", test_targets.shape)    # 102,

## 데이터 표준화
from sklearn.preprocessing import StandardScaler
# mean = train_data.mean(axis=0)
# std = train_data.std(axis=0)
# train_data -= mean
# train_data /= std

# test_data -= mean
# test_data /= std
scaler = StandardScaler()
scaler.fit(train_data)
train_data_sc = scaler.transform(train_data)
test_data_sc = scaler.transform(test_data)



from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"]) # mae >> 절댓값

    return model


# from sklearn.model_selection import StratifiedKFold
seed=77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score

model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=1, verbose=1)
Kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, train_data, train_targets, cv=Kfold)


import numpy as np

print(results)
print(np.mean(results))