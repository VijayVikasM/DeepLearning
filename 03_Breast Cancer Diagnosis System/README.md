
# Breast Cancer Diagnosis System

This project provides a neural network-based tool to classify breast cancer tumors as *benign* or *malignant* based on diagnostic data. The system is designed for both training and real-time diagnosis using a pre-trained model.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Evaluating the Model](#2-evaluating-the-model)
  - [3. Making Predictions](#3-making-predictions)
- [Project Modules](#project-modules)
- [Model Details](#model-details)
- [Results Visualization](#results-visualization)
- [Future Improvements](#future-improvements)

---

## Overview

The Breast Cancer Diagnosis System classifies breast cancer tumors as benign or malignant using diagnostic features from a widely-used dataset. The system employs a neural network model implemented with TensorFlow/Keras. This project is modular and organized, allowing for ease of use, flexibility in model training, evaluation, and making real-time predictions.

---

## Project Structure

```plaintext
breast_cancer_diagnosis_system/
├── data/                       # Folder for datasets, if applicable
├── src/
│   ├── data_loader.py          # For loading and preprocessing data
│   ├── model.py                # For creating and training the model
│   ├── evaluator.py            # For model evaluation
│   ├── predictor.py            # For making predictions on new data
│   └── visualizer.py           # For visualizing training results
├── scripts/
│   └── run_diagnosis.py        # Main script for training, evaluating, and diagnosing
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview and instructions
```

---

## Requirements

To run this project, you need:

- Python 3.7+
- Packages listed in `requirements.txt`:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/breast_cancer_diagnosis_system.git
   cd breast_cancer_diagnosis_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the diagnosis script**:
   ```bash
   python scripts/run_diagnosis.py
   ```

---

## Usage

The project is designed to run through a single entry-point script, `run_diagnosis.py`, which provides an end-to-end solution for data loading, model training, evaluation, and prediction.

### 1. Training the Model

To train the neural network, the script loads and preprocesses the data, initializes the model, and trains it on the training set. The trained model is saved for future use.

### 2. Evaluating the Model

The script evaluates the trained model on the test set, providing metrics like accuracy and loss, and generates a classification report with detailed performance metrics.

### 3. Making Predictions

The system allows for real-time predictions on new input data by loading the trained model, standardizing the input, and outputting a diagnosis (benign or malignant). The script demonstrates this with an example input in `run_diagnosis.py`.

---

## Project Modules

### `src/data_loader.py`

Handles loading the breast cancer dataset from `sklearn`, preprocessing it, and splitting it into training and testing sets. Also standardizes the input data for better model performance.

### `src/model.py`

Defines the neural network model with TensorFlow/Keras. The model consists of three layers:
- Input layer for accepting 30 diagnostic features
- Hidden layer with ReLU activation
- Output layer with two nodes and sigmoid activation (binary classification).

### `src/evaluator.py`

Contains functions for evaluating the model on test data. It calculates accuracy, loss, and provides a classification report with detailed performance metrics.

### `src/predictor.py`

Allows for predictions on new data input. Given a set of diagnostic measurements, it predicts if a tumor is benign or malignant based on the trained model.

### `src/visualizer.py`

Visualizes model training history, including accuracy and loss over epochs, for better understanding of model performance.

### `scripts/run_diagnosis.py`

The main script to run the full workflow, integrating all modules. It loads the data, trains the model, evaluates it, visualizes the results, and performs a sample prediction.

---

## Model Details

The neural network model is a simple feedforward architecture designed for binary classification:
- **Input Layer**: Accepts 30 diagnostic features.
- **Hidden Layer**: 20 neurons with ReLU activation.
- **Output Layer**: 2 neurons with sigmoid activation for benign and malignant classifications.

---

## Results Visualization

The training process is visualized with accuracy and loss plots, showing both training and validation performance over epochs. This helps in understanding if the model is overfitting, underfitting, or performing well.

---

## Future Improvements

Potential areas for enhancement:
- Experimenting with different neural network architectures.
- Adding hyperparameter tuning for optimized performance.
- Exploring additional machine learning algorithms for comparison.

---

This project provides a modular, efficient, and scalable approach to breast cancer diagnosis using machine learning, suitable for extending and adapting in various diagnostic contexts.
