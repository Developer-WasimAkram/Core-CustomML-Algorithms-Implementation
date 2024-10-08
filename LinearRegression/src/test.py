#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linearRegression import Linear_Regression
from utils import mean_squared_error, r2_score


x, y = datasets.make_regression( n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=1234)


regressor = Linear_Regression(learning_rate=0.01, num_iterations=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

accu = r2_score(y_test, predictions)
print("Accuracy:", accu)
print("_"*50)
print("Validate with scikit learn linear regression model)")


from sklearn.linear_model import LinearRegression as sk_LinearRegression
sk_regressor =sk_LinearRegression()
sk_regressor.fit(X_train, y_train)
sk_predictions = sk_regressor.predict(X_test)

mse = mean_squared_error(y_test, sk_predictions)
print("MSE:", mse)

accu = r2_score(y_test, sk_predictions)
print("Accuracy:", accu)

