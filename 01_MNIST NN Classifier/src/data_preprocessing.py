import numpy as np
from tensorflow import keras

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    return X_train, y_train, X_test, y_test
