#(3) example03

# knn visualization example, k = 1
# classification
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
# regression
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

# knn visualization example, k = 3
# classification
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()
# regression
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

## using knn model forge dataset for classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# dataset load
X, y = mglearn.datasets.make_forge()
# split train, test 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# define model, k = 3
knn = KNeighborsClassifier(n_neighbors=3)
# input train data
knn.fit(X_train, y_train)
# accuracy
acc = knn.score(X_test, y_test)
print('accurarcy : {}'.format(acc))

# visualization
fig, axes = plt.subplots(1, 3, figsize=(10,3))

for n_neighbors, ax in zip([1, 3, 9], axes):
   knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
   mglearn.plots.plot_2d_separator(knn, X, fill=True, eps=0.5, ax=ax, alpha=.4)
   mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
   ax.set_title('{} neighbor'.format(n_neighbors))
   ax.set_xlabel('feature 0')
   ax.set_ylabel('feature 1')
axes[0].legend(loc=3)
plt.show()

## using knn model breast cancer dataset for classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# dataset load
cancer = load_breast_cancer()
# split train, test 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# define neighbors
neighbors_settings = range(1,11)
# define model, k = 3, and prediction
for n_neighbors in neighbors_settings:
   knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
   training_accuracy.append(knn.score(X_train,y_train))
   test_accuracy.append(knn.score(X_test,y_test))
# visualization
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

## using knn model make wave dataset for regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
# data load
X, y = mglearn.datasets.make_wave(n_samples=40)
# split train, test 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# define model, k = 3
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,y_train)
# accuracy
acc = knn.score(X_test,y_test)
print('accuracy : {}'.format(acc))
# visualization
fig, axes = plt.subplots(1, 3, figsize=(15,4))

line = np.linspace(-3, 3, 1000).reshape(-1,1)
for n_neighbors, ax in zip([1, 3, 9], axes):
   knn = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train,y_train)
   ax.plot(line, knn.predict(line))
   ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
   ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
   ax.set_title('{} neighbor training score: {:.2f}, test score: {:.2f}'.format(n_neighbors, knn.score(X_train, y_train),knn.score(X_test, y_test)))
   ax.set_xlabel('feature')
   ax.set_ylabel('target')
axes[0].legend(['model predict','training data/target','test data/target'],loc='best')
plt.show()
