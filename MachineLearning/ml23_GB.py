from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# tree = DecisionTreeClassifier(random_state=0)
# tree = RandomForestClassifier(random_state=0)
# tree = GradientBoostingClassifier(random_state=0)
# tree.fit(x_train, y_train)
# print("훈련 세트 정확도 >> {:.3f}".format(tree.score(x_train, y_train)))  # 1.00
# print("테스트 세트 정확도 >> {:.3f}".format(tree.score(x_test, y_test)))  # 0.937



# tree = DecisionTreeClassifier(max_depth=7, random_state=0)      # max_depth >> 1.00 / 0.937
tree = GradientBoostingClassifier(max_depth=7, random_state=0)    # max_depth >> 1.00 / 0.937
tree.fit(x_train, y_train)
print("훈련 세트 정확도 >> {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도 >> {:.3f}".format(tree.score(x_test, y_test)))

## max_depth >> 깊이

# 데이터 컬럼 별 중요도 출력
print("특성 중요도\n", tree.feature_importances_)


# 중요도 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)


plot_feature_importances_cancer(tree)
plt.show()