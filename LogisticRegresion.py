from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils import plot_model

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Load the dataset from 'tae.dat' file
dataset = np.loadtxt('tae.dat', delimiter=',')
X = dataset[:, 0:-1]  # Input features
y = dataset[:, -1]  # Target variable

# Normalize the input features using MinMaxScaler
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)

# Create a logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Predict the classes for the testing data
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the logistic regression model
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy))

# Calculate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot()
plt.title('Confusion Matrix')
plt.show()
