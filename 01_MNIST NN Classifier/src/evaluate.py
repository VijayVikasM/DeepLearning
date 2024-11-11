from tensorflow import keras
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(X_test, y_test):
    model = keras.models.load_model('model.h5')
    y_predicted = model.predict(X_test)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    
    # Generate and display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    _, _, X_test, y_test = load_and_preprocess_data()
    evaluate_model(X_test, y_test)
