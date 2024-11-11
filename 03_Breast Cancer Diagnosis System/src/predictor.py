
import numpy as np
import tensorflow as tf

def predict(input_data, scaler):
    model = tf.keras.models.load_model("breast_cancer_model.h5")
    input_data = scaler.transform(np.asarray(input_data).reshape(1, -1))
    prediction = model.predict(input_data)
    prediction_label = np.argmax(prediction)
    if prediction_label == 0:
        print("The tumor is Malignant")
    else:
        print("The tumor is Benign")
