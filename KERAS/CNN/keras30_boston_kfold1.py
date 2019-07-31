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



import numpy as np

k = 4
num_val_samples = len(train_data) // k      # 404/4
num_epochs = 1
all_scores = []

for i in range(k):
    print("처리중인 폴드 #", i)
    # # 검증데이터 준비 : k번째 분할
    # val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    # val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    # print("val_data shape : ", val_data.shape)     # [0:101, 13] > [101:202, 13] > [202:303, 13] > [303:404, 13]
    # print("val_targets shape : ", val_targets.shape)  # [0:101, ] > [101:202, ] > [202:303, ] >  [303:404, ]



    # # 훈련데이터 준비 : 다른분할 전체 >>
    # partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]], axis=0)
    # partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i+1) * num_val_samples:]], axis=0)
    # print("partial_train_data shape : ", partial_train_data.shape)
    # print("partial_train_targets shape : ", partial_train_targets.shape)
    
    ##########################################################################################################################################
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

    model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=1, verbose=1)
    kfold = KFold(n_splits=k, shuffle=True, random_state=77)
    score = cross_val_score(model, train_data, train_targets, cv=kfold)

    print(score)
    print(np.mean(score))

'''
    # 케라스 모델 구성
    model = build_model()


    # 모델 훈련
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)


    ## 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


print(all_scores)
print(np.mean(all_scores))  # >> numpy 리스트 평균값 출력

'''