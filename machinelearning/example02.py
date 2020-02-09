#(2) example02

# forge dataset
import mglearn
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.legend(["class 0","class 1"], loc=4)
plt.xlabel("first feature")
plt.ylabel("second feature")
plt.show()

# wave dataset
import mglearn
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("feature")
plt.ylabel("target")
plt.show()

# breast cancer dataset
from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()
print('(data points, features) = {}'.format(cancer['data'].shape))
print("{}".format({n: v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("{}".format(cancer.feature_names))
plt.plot(cancer['data'],cancer['target'],'o')
plt.xlabel("feature")
plt.ylabel("target")
plt.show()

# boston housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
print('(data points, features) = {}'.format(boston['data'].shape))
print("{}".format(boston.feature_names))
plt.plot(boston['data'],boston['target'],'o')
plt.xlabel("feature")
plt.ylabel("target")
plt.show()

import mglearn
X, y = mglearn.datasets.load_extended_boston()
plt.plot(X,y,'o')
plt.xlabel("feature")
plt.ylabel("target")
plt.show()


