from sklearn.linear_model import LogisticRegression, Perceptron
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

# Create a Perceptron model
pn = Perceptron(tol=1e-3, random_state=0)

# Fit the Perceptron model to the training data
pn.fit(X_train, y_train)

# Calculate the training and testing scores of the Perceptron model
train_score = pn.score(X_train, y_train)
test_score = pn.score(X_test, y_test)
print('Perceptron Training Score:', train_score)
print('Perceptron Test Score:', test_score)

# Predict the classes for the testing data using the Perceptron model
y_pred = pn.predict(X_test)

# Calculate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
