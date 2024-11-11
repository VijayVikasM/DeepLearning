import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from data_preprocessing import load_and_preprocess_data

# Load the trained model and data
model = load_model('fashion_mnist_model.h5')
_, _, X_test, y_test = load_and_preprocess_data()

# Define class labels for Fashion MNIST dataset
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Predict on the test set
y_predicted = model.predict(X_test)

# Function to display images with labels
def display_images(images, true_labels, predicted_labels, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        true_label = class_labels[true_labels[i]]
        predicted_label = class_labels[np.argmax(predicted_labels[i])]
        plt.title(f"T: {true_label} P: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_predictions.png")  # Save the plot to a file


# Display the first 10 test images with their true and predicted labels
display_images(X_test, y_test, y_predicted, num_images=10)
