#(1) example01

# numpy example
import numpy as np
x = np.array([1,2,3])
print("x : {}".format(x))

# sciPy example
import numpy as np
from scipy import sparse
eye = np.eye(10)
print("numpy array : {}".format(eye))

import numpy as np
from scipy import sparse
data = np.ones(10)
row = np.arange(10)
col = np.range(10)
eye_coo = sparse.coo_matrix((data,(row,col)))
print("convert to coo matrix : {}".format(eye_coo))

# matplotlib example
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker="x")
plt.show()

# pandas example
import pandas as pd
data = {'name': ["jaeyun","jaelong"], 'age' : [26,4]}
data_pandas = pd.DataFrame(data)
print(data_pandas)

# iris example
# data load
from sklearn.datasets import load_iris
iris_dataset = load_iris()
# check data
print(iris_dataset['data'])
# check label
print(iris_dataset['target'])

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset[
'target'], random_state=0)

# data visualization
# comvert numpy to DataFrame
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
plt.show()

# define model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
# predict
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("result : {}".format(prediction))

# accuracy
y_pred = knn.predict(X_test)
acc_1 = np.mean(y_pred == y_test)
acc_2 = knn.score(X_test,y_test)