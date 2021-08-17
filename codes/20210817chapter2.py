import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

#%%
# データセットの生成
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

print("X shape: ", X.shape)

# FutureWarning: Function make_blobs is deprecated; Please import make_blobs directly from scikit-learn
# warnings.warn(msg, category=FutureWarning)

#%%
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

#%%
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n", cancer.keys())

#%%
print("Shape of cancer data: \n", cancer.data.shape)

#%%
print("Sample counts per class: \n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

#%%
print("Feature names: \n", cancer.feature_names)

#%%
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: ", boston.data.shape)

#%%
X, y = mglearn.datasets.load_extended_boston()
print("X.shape", X.shape)



