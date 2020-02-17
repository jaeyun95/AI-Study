#(4) example04

# show Linear Regression result
import matplotlib.pyplot as plt
import mglearn

mglearn.plots.plot_linear_regression_wave()
plt.show()

# method of least squares
# for wave dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef :{}".format(lr.coef_))
print("lr.intercept :{}".format(lr.intercept_))

print("training set score : {:.2f}".format(lr.score(X_train, y_train)))
print("test set score : {:.2f}".format(lr.score(X_test, y_test)))

# method of least squares
# for boston housing dataset
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef :{}".format(lr.coef_))
print("lr.intercept :{}".format(lr.intercept_))

print("training set score : {:.2f}".format(lr.score(X_train, y_train)))
print("test set score : {:.2f}".format(lr.score(X_test, y_test)))

# Ridge Regression
# for boston housing dataset
# alpha is 1.0
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)

print("lr.coef :{}".format(ridge.coef_))
print("lr.intercept :{}".format(ridge.intercept_))

print("training set score : {:.2f}".format(ridge.score(X_train, y_train)))
print("test set score : {:.2f}".format(ridge.score(X_test, y_test)))

# alpha is 10
ridge10 = Ridge(alpha=10).fit(X_train, y_train)

print("training set score : {:.2f}".format(ridge10.score(X_train, y_train)))
print("test set score : {:.2f}".format(ridge10.score(X_test, y_test)))

# alpha is 0.1
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

print("training set score : {:.2f}".format(ridge01.score(X_train, y_train)))
print("test set score : {:.2f}".format(ridge01.score(X_test, y_test)))

# visualization
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, '^', label="Ridge alpha=1.0")
plt.plot(ridge01.coef_, '^', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")

plt.xlabel("coef list")
plt.ylabel("coef size")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()

# Lasso Regression
# for boston housing dataset
# alpha is 1.0
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)

print("training set score : {:.2f}".format(lasso.score(X_train, y_train)))
print("test set score : {:.2f}".format(lasso.score(X_test, y_test)))
print("used features : {}".format(np.sum(lasso.coef_ != 0)))

# alpha is 0.01
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

print("training set score : {:.2f}".format(lasso001.score(X_train, y_train)))
print("test set score : {:.2f}".format(lasso001.score(X_test, y_test)))
print("used features : {}".format(np.sum(lasso001.coef_ != 0)))

# alpha is 0.0001
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)

print("training set score : {:.2f}".format(lasso00001.score(X_train, y_train)))
print("test set score : {:.2f}".format(lasso00001.score(X_test, y_test)))
print("used features : {}".format(np.sum(lasso00001.coef_ != 0)))

# visualization
plt.plot(lasso.coef_, '^', label="Lasso alpha=1.0")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, '^', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, '^', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")

plt.xlabel("coef list")
plt.ylabel("coef size")
plt.ylim(-25,25)
plt.legend(ncol=3, loc=(0,1.05))
plt.show()


