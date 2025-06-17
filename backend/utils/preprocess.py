import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess(data, feature_col: str = "Close", window_size: int = 60):
    """
    Accepts either a DataFrame or a CSV path, normalizes feature column,
    and creates sliding windows for LSTM input.

    Returns:
        Tuple of (X, y) NumPy arrays ready for LSTM.
        Each sample in X is a window of 60 past days of normalized closing prices.
        Each label in y is the next normalized price right after the corresponding window.
    """
    if isinstance(data, str):
        df = pd.read_csv(data, index_col="Date", parse_dates=True)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a DataFrame or a CSV file path")

    values = df[[feature_col]].values

    scaler = MinMaxScaler(feature_range=(0, 1)) # scale data to [0, 1]
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler