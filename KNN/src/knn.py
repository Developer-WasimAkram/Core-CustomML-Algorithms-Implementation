from utils import euclidean_distance 
import numpy as np



class KNN:
        """
        A class used to represent a K-Nearest Neighbors (KNN) classifier.
    
        ...
    
        Attributes
        ----------
        k : int
            The number of nearest neighbors to consider for classification.
    
        Methods
        -------
        __init__(self, k):
            Constructs all the necessary attributes for the KNN object.
    
            Parameters
            ----------
            k : int
                The number of nearest neighbors to consider for classification.
        """
        def __init__(self, k=3):
            self.k = k
            
            
        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            
            
        def predict(self, X_test):
            
            y_pred = [self._predict(x) for x in X_test]
            return np.array(y_pred)
            
        def _predict(self, x):
            """
            Predicts the class label for a given input data point using the K-Nearest Neighbors algorithm.

            Parameters:
            x (numpy.ndarray): A single data point represented as a 1D numpy array.

            Returns:
            int: The predicted class label for the input data point.
            """
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in nearest_indices]
            most_common_label = np.argmax(np.bincount(k_nearest_labels))
            return most_common_label
            
            
            
            
            
            
            
            
            
            
            
            
            