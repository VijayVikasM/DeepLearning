
import tensorflow as tf
from tensorflow import keras

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(30,)),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train):
    history = model.fit(X_train, Y_train, validation_split=0.1, epochs=10)
    model.save("breast_cancer_model.h5")
    return history
