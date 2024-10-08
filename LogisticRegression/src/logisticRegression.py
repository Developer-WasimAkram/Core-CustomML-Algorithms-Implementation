import numpy as np 

import numpy as np


class Logistic_Regression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model with given learning rate and number of iterations.

        Parameters:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        num_iterations (int): The number of iterations for gradient descent. Default is 1000.

        Returns:
        None
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.

        Parameters:
        X (numpy.ndarray): A 2D array of shape (n_samples, n_features) representing the input features.
        y (numpy.ndarray): A 1D array of shape (n_samples,) representing the target labels.

        Returns:
        None
        """
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):

            # Compute linear regression predictions
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def predict(self, X):
        """
        Predict class labels for input features using the trained logistic regression model.

        Parameters:
        X (numpy.ndarray): A 2D array of shape (n_samples, n_features) representing the input features.

        Returns:
        numpy.ndarray: A 1D array of shape (n_samples,) containing the predicted class labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def _sigmoid(self, x):
        """
        Compute the sigmoid function for the given input.

        The sigmoid function is a mathematical function that maps any real number to a value between 0 and 1.
        It is commonly used in binary classification problems, where the output can be interpreted as the probability of a sample belonging to a particular class.

        Parameters:
        x (numpy.ndarray or float): The input value(s) for which the sigmoid function needs to be computed.

        Returns:
        numpy.ndarray or float: The sigmoid value(s) corresponding to the input(s).
        """
        return 1 / (1 + np.exp(-x))
# Example usage TestModel

