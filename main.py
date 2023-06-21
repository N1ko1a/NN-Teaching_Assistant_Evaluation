import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils import plot_model

scaler = MinMaxScaler()

# Load and preprocess the dataset
dataset = np.loadtxt('tae.dat', delimiter=',')
X = dataset[:, 0:-1]
Y = dataset[:, -1]

# Normalize the input features
X_normalized = scaler.fit_transform(X)

# Encode the target variable as integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_encoded, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_normalized, Y_encoded, test_size=0.1, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(np.max(Y_encoded) + 1, activation='softmax'))

# Set the learning rate for the Adam optimizer
learning_rate = 0.003 
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the specified optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=5, validation_data=(X_valid, y_valid))

# Plot accuracy during training
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict classes for the normalized dataset
y_pred = model.predict(X_test)
y_pred_decoded = np.argmax(y_pred, axis=1)

# Compute accuracy and display confusion matrix
accuracy = accuracy_score(y_test, y_pred_decoded)
cm = confusion_matrix(y_test, y_pred_decoded)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
labels = label_encoder.classes_
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
