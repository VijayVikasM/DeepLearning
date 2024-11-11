from data_preprocessing import load_and_preprocess_data
from model import create_model



def train_model():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    model = create_model()
    model.fit(X_train, y_train, epochs=5)
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    model.save("model.keras", save_format="keras")


