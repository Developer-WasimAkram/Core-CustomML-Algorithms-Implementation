import numpy as np   


def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points in a multi-dimensional space.
    Parameters:
    x1 (numpy.ndarray): A 1D numpy array representing the coordinates of the first point.
    x2 (numpy.ndarray): A 1D numpy array representing the coordinates of the second point.
    Returns:
    float: The Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x2 - x1) ** 2))

