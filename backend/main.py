import torch
import torch.nn as nn
import torch.optim as optim
from data.fetch_data import download_stock_data
from utils.preprocess import preprocess
from models.lstm_model import LSTMModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
WINDOW_SIZE = 60
HIDDEN_SIZE = 50
NUM_LAYERS = 2
DROPOUT = 0.0
MODEL_SAVE_PATH = "backend/models/trained_lstm_model.pth"

def prepare_data():
    """
    Downloads and preprocesses stock data for a given ticker.
    Splits the preprocessed data into training and testing sets.

    Returns:
        X_train, X_test: torch.Tensor of input sequences
        y_train, y_test: torch.Tensor of target values
        scaler: fitted scaler for inverse transforming outputs
    """ 
    df = download_stock_data("AAPL", "2020-01-01", "2023-01-01")
    X, y, scaler = preprocess(df, window_size=WINDOW_SIZE)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # split into training and testiing sets (80% train, 20% test)
    split = int(0.8 * len(X_tensor))
    return (
        X_tensor[:split], X_tensor[split:],
        y_tensor[:split], y_tensor[split:],
        scaler
    )

def train_model(model, X_train, y_train):
    """
    Trains the LSTM model using MSE loss and Adam optimizer.

    Args:
        model: LSTMModel instance
        X_train: training input tensor
        y_train: training target tensor
    """
    criterion = nn.MSELoss() # define MSE loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # define regularized Adam optimizer
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        permutation = torch.randperm(X_train.size(0)) # shuffle training data each epoch
        for i in range(0, X_train.size(0), BATCH_SIZE):
            # fetch mini-batch
            indices = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # backpropagation and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # peinr acumulated average loss for the epoch
        avg_loss = total_loss / (len(X_train) // BATCH_SIZE)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.6f}")

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluates the trained model on test data and prints performance metrics.

    Args:
        model: trained LSTM model
        X_test: test input tensor
        y_test: test target tensor
        scaler: fitted scaler for inverse transforming outputs

    Returns:
        predictions_np: inverse transformed model predictions
        actuals_np: inverse transformed true values
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        
        # inverse transform predictions and actuals
        predictions_np = scaler.inverse_transform(predictions.numpy())
        actuals_np = scaler.inverse_transform(y_test.numpy())

        mse = mean_squared_error(actuals_np, predictions_np)
        mae = mean_absolute_error(actuals_np, predictions_np)
        r2 = r2_score(actuals_np, predictions_np)
        variance = np.var(predictions_np - actuals_np)
        std_dev = np.std(predictions_np - actuals_np)

        print(f"\nTest MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R^2 Score: {r2:.4f}")
        print(f"Prediction Error Variance: {variance:.4f}")
        print(f"Prediction Error Std Dev: {std_dev:.4f}")
        return predictions_np, actuals_np

def save_model(model, path=MODEL_SAVE_PATH):
    """
    Saves the trained PyTorch model to disk.

    Args:
        model: trained LSTM model
        path: file path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to {path}")

def main():
    """
    Main pipeline for training, evaluating, and saving the LSTM model.
    """
    
    # load and split data
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # initialize model
    model = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    
    # train model 
    train_model(model, X_train, y_train)
    
    # evaluate on test set
    evaluate_model(model, X_test, y_test, scaler)
    
    # save model to disk
    save_model(model)

if __name__ == "__main__":
    main()
