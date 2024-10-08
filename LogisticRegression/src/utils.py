import numpy as np 
def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of a classification model.

    Parameters:
    y_true (numpy.ndarray): The true labels of the samples.
    y_pred (numpy.ndarray): The predicted labels of the samples.

    Returns:
    float: The accuracy of the model, calculated as the number of correctly predicted samples divided by the total number of samples.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy