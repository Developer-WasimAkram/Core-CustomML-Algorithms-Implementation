from logisticRegression import Logistic_Regression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load iris dataset
bc = datasets.load_iris()
X,y = bc.data, bc.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a logistic regression classifier
clf = Logistic_Regression(learning_rate=0.01, num_iterations=10000)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels of test set
 
y_pred = clf.predict(X_test)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"Accuracy from implemented algorithm _Accuracy: 0.7"

# Compare the accuracy with sklearn's logistic regression
clf_sk = LogisticRegression()
clf_sk.fit(X_train, y_train)
y_pred_sk = clf_sk.predict(X_test)
accuracy_sk = accuracy_score(y_test, y_pred_sk)
print("Accuracy from sklearn's logistic regression:", accuracy_sk)
"Accuracy from sklearn's logistic regression _Accuracy from sklearn's logistic regression: 0.7"