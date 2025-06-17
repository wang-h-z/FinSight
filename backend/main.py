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
WINDOW_SIZE = 60
HIDDEN_SIZE = 50
WEIGHT_DECAY = 1e-5
DROPOUT = 0
NUM_LAYERS = 2
MODEL_SAVE_PATH = "backend/models/trained_lstm_model.pth"

# load and preprocess data
df = download_stock_data("AAPL", "2020-01-01", "2023-01-01")
X, y, scaler = preprocess(df, window_size=WINDOW_SIZE)

# convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# train/test split (80/20)
split = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

# model, loss, optimizer
model = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(X_train) // BATCH_SIZE)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.6f}")

# evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
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

# save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")