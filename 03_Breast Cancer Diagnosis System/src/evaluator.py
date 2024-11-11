
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf

def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {accuracy}")
    Y_pred = model.predict(X_test)
    Y_pred_labels = [np.argmax(i) for i in Y_pred]
    print(classification_report(Y_test, Y_pred_labels))
