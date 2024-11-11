
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data
from src.model import create_model, train_model
from src.evaluator import evaluate_model
from src.predictor import predict
from src.visualizer import plot_history



def main():
    # Load data
    X_train, X_test, Y_train, Y_test, scaler = load_data()

    # Create and train model
    model = create_model()
    history = train_model(model, X_train, Y_train)

    # Visualize training history
    plot_history(history)

    # Evaluate model
    evaluate_model(model, X_test, Y_test)

    # Predict new input
    input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
                  0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
                  12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)
    predict(input_data, scaler)

if __name__ == "__main__":
    main()
