import unittest
from src.model import create_model
from src.data_preprocessing import load_and_preprocess_data

class TestDigitRecognition(unittest.TestCase):
    def test_data_loading(self):
        X_train, y_train, X_test, y_test = load_and_preprocess_data()
        self.assertEqual(X_train.shape[1], 784)
        self.assertEqual(len(y_train), len(X_train))

    def test_model_creation(self):
        model = create_model()
        self.assertEqual(len(model.layers), 1)  # Check for single dense layer

if __name__ == "__main__":
    unittest.main()
