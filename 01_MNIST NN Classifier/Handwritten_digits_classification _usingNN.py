# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Display the shape of the training set
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Visualize the first sample in the training dataset
plt.matshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Normalize the datasets
# This is done to scale pixel values between 0 and 1 for better model performance
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the datasets
# Each image is reshaped from 28x28 to a single 784-element array
X_train_flattened = X_train.reshape(len(X_train), 28 * 28)
X_test_flattened = X_test.reshape(len(X_test), 28 * 28)

# Define a neural network model
# We use a simple sequential model with one Dense layer and sigmoid activation
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

# Compile the model with necessary parameters
# 'adam' optimizer is used for efficient gradient descent
# 'sparse_categorical_crossentropy' loss is used for multi-class classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
# We run for 5 epochs
model.fit(X_train_flattened, y_train, epochs=5)

# Evaluate the model on test data to check its performance
test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test)
print("Test accuracy:", test_accuracy)

# Make predictions on the test set
y_predicted = model.predict(X_test_flattened)

# Example of viewing prediction for the first test sample
plt.matshow(X_test[0], cmap='gray')
plt.title(f"Predicted Label: {np.argmax(y_predicted[0])}")
plt.show()

# Convert predictions to label format by taking the index of maximum value in predictions
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Plotting the confusion matrix to evaluate performance
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix for test labels vs. predicted labels
conf_matrix = confusion_matrix(y_test, y_predicted_labels)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
