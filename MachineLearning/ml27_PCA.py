from sklearn.datasets import  load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
sc = StandardScaler()

X_sc = sc.fit_transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_sc)


X_pca = pca.transform(X_sc)
print("원본 상태 >> ", X_sc.shape)
print("축소된 상태 >> ", X_pca.shape)


'''

components=2
원본 상태 >>  (569, 30)
축소된 상태 >>  (569, 2)

components=13
원본 상태 >>  (569, 30)
축소된 상태 >>  (569, 13)
'''