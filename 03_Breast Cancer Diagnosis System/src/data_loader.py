
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.datasets

def load_data():
    # Load data from sklearn
    breast_cancer_data = sklearn.datasets.load_breast_cancer()
    data_frame = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
    data_frame['label'] = breast_cancer_data.target
    
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']
    
    # Split and scale data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test, scaler
