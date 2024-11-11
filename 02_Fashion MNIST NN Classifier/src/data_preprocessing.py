import numpy as np
import keras


def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values
    return X_train, y_train, X_test, y_test
