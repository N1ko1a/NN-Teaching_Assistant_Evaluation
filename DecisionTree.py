from lib2to3.pgen2.pgen import DFAState
import numpy as np
from six import StringIO
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
from matplotlib import pyplot as plt
import pydotplus

# Read the dataset from the 'tae.csv' file
df = pd.read_csv("tae.dat")

# Separate the input features (X) and the target variable (y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1:]

# Normalize the input features
normalized_X = preprocessing.normalize(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the training data
clf = clf.fit(X_train, y_train)

# Predict the classes for the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate the visualization of the decision tree
dot_data = StringIO()
export_graphviz(
    clf,
    out_file=dot_data,
    feature_names=X.columns,
    class_names=y.iloc[:, 0].astype(str).unique(),
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("decision_tree.png")
