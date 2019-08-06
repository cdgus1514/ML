from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


boston = load_boston()

print("data shape >> ", boston.data.shape)      # (506, 13)
print(boston.keys())
print(boston.target)
print("target shape >>", boston.target.shape)   # (506,)

x = boston.data
y = boston.target
# print(type(boston))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


## 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit_transform(x_train)
sc.transform(x_test)


from sklearn.linear_model import LinearRegression, Ridge, Lasso   #Ridge, Lasso 모델로 완성
# model = LinearRegression(Ridge)   #
model = LinearRegression(Lasso)     # 0.68
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# y_pred = model.predict(x_test)
print("score >> ", score)