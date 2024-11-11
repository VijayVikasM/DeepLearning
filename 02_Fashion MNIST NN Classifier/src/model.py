from keras.models import Sequential
from keras.layers import Flatten, Dense

def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(100, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
