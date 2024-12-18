import numpy as np
class Linear_Regression:
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples,n_features = X.shape

        # Initialize weights and bias
        self.weights =np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):

            # Compute linear regression predictions
            y_predicted = np.dot(X,self.weights) + self.bias

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T,(y_predicted-y))
            db=((1/n_samples) * np.sum(y_predicted-y))

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,X):
        y_predicted = np.dot(X,self.weights) + self.bias
        return y_predicted
    

# Example usage TestModel

