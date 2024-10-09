
from knn import KNN 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from utils import euclidean_distance
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Load iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Print classification accuracy
    print("KNN classification accuracy:", accuracy_score(y_test, predictions))