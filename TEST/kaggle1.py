import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd


train = pd.read_csv("C:/Users/bitcamp/Downloads/cat-in-the-dat/train.csv")
test = pd.read_csv("C:/Users/bitcamp/Downloads/cat-in-the-dat/test.csv")

Combined_data = pd.concat([train.drop(['target'], axis=1), test], axis=0, ignore_index=True)
print('Shape of training dataset: {}'.format(train.shape))
print('Shape of testing dataset: {}'.format(test.shape))
print('Shape of combined dataset: {}'.format(Combined_data.shape))


## check for missing data
# total = Combined_data.isnull().sum().sort_values(ascending=False)
# percent = (Combined_data.isnull().sum())/Combined_data.isnull().count().sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
# missing_data.head(5)



X_train = train.drop(["target"], axis=1)
Y_train = train["target"]
print(X_train.shape)    # (300000, 24)
print(Y_train.shape)    # (300000,)


bin_cats = X_train[["bin_0", "bin_1", "bin_2", "bin_3", "bin_4"]]
bin_cats.bin_3 = 1*(bin_cats.bin_3 == "T")
bin_cats.bin_4 = 1*(bin_cats.bin_4 == "Y")


from sklearn.preprocessing import OneHotEncoder

nom_cats = X_train[["nom_1", "nom_2", "nom_3", "nom_4", "nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]]
sc = OneHotEncoder()
nom_cats = sc.fit_transform(nom_cats)
nom_cats


from pandas.api.types import CategoricalDtype

ord_cat_names = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]
ord_cats = X_train[ord_cat_names]
ord_cats.ord_1 = pd.factorize(ord_cats.ord_1.astype(CategoricalDtype(categories=["Novice", "Contibutor", "Expert", "Master", "Grandmaster"], ordered=True)))[0]
ord_cats.ord_2 = pd.factorize(ord_cats.ord_2.astype(CategoricalDtype(categories=["Freezing", "Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"], ordered=True)))[0]
ord_cats.ord_3 = ord_cats["ord_3"].apply(lambda x: ord(x)-ord("a"))
ord_cats.ord_4 = ord_cats["ord_4"].apply(lambda x: ord(x)-ord("A"))


import string

d = {}
for i, s in enumerate(string.ascii_letters):
    d[s] = i

ord_cats.ord_5 = ord_cats["ord_5"].apply(lambda x: 10*d[x[0]]+d[x[1]])


from scipy import sparse

cyclic_cats = X_train[["day", "month"]]
X = np.concatenate((bin_cats.values, ord_cats.values), axis=1)
X = sparse.csr_matrix(X)
X = sparse.hstack((X, nom_cats))

Y = train["target"]
print('The original input representation has shape {}'.format(train.shape))
print('The one-hot encoded input representation has shape {}'.format(X.shape))



## data split
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print('The shape of the training input is {}'.format(X_train.shape))    # (240000, 16193)
print('The shape of the validation input is {}'.format(X_val.shape))    # (60000, 16193)
print('The shape of the training output is {}'.format(Y_train.shape))   # (240000,)
print('The shape of the validation output is {}'.format(Y_val.shape))   # (60000,)




## model
from sklearn.ensemble import AdaBoostClassifier

def adaboost(X_train, X_val, Y_train):
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_val)

    return y_pred

y_baseline = adaboost(X_train, X_val, Y_train)



## SMOTE
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train)
y_smote = adaboost(X_train_sm, X_val, Y_train_sm)
'''

## RUS
from sklearn.utils import resample
from sklearn.metrics import classification_report

X_full = X_train.copy()
X_full["target"] = Y_train
X_maj = X_full[X_full.target == 0]
X_min = X_full[X_full.target == 1]
X_maj_rus = resample(X_maj, replace=False, n_samples=len(X_min), random_state=44)
X_rus = pd.concat([X_maj_rus, X_min])

X_train_rus = X_rus.drop(["target"], axis=1)
Y_train_rus = X_rus.target

Y_rus = adaboost(X_train_rus, X_val, Y_train_rus)
print("RUS Adaboost")
print(classification_report(Y_rus, Y_val))




### result

print("Vanilla AdaBoost")
print(classification_report(y_baseline, Y_val))

print("SMOTE AdaBoost")
print(classification_report(y_smote, Y_val))

print("RUS Adaboost")
print(classification_report(Y_rus, Y_val))

'''


# 지표 >> ROC