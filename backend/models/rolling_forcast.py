import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def improved_rolling_forecast(model, X_test, y_test, scaler, 
                            retrain_freq=20, confidence_threshold=0.1,
                            use_actual_periodically=True, actual_freq=10):
    """
    Improved rolling forecast with multiple strategies to prevent drift
    
    Args:
        model: trained PyTorch model
        X_test: test sequences
        y_test: actual test values (for comparison and correction)
        scaler: fitted scaler for inverse transform
        retrain_freq: how often to retrain model (epochs)
        confidence_threshold: threshold for prediction confidence
        use_actual_periodically: whether to inject actual values periodically
        actual_freq: frequency of actual value injection
    """
    rolling_preds = []
    input_seq = X_test[0].unsqueeze(0).clone()  # initial window
    
    # Track prediction errors for adaptive correction
    recent_errors = []
    error_window = 10
    
    for i in range(len(X_test)):
        with torch.no_grad():
            pred = model(input_seq)
            
        # Apply error correction based on recent performance
        if len(recent_errors) >= 3:
            error_trend = np.mean(recent_errors[-3:])
            # Correct for systematic bias
            pred_corrected = pred.item() - error_trend * 0.3
        else:
            pred_corrected = pred.item()
            
        rolling_preds.append(pred_corrected)
        
        # Calculate prediction error for tracking
        if i < len(y_test):
            actual_scaled = y_test[i].item()
            pred_error = actual_scaled - pred.item()
            recent_errors.append(pred_error)
            if len(recent_errors) > error_window:
                recent_errors.pop(0)
        
        # Strategy 1: Periodically inject actual values to prevent drift
        if use_actual_periodically and i % actual_freq == 0 and i < len(X_test) - 1:
            # Use actual value instead of prediction
            next_step = X_test[i + 1][:, -1:, :]  # last value of next sequence
        else:
            # Use prediction
            next_step = torch.tensor(pred_corrected).reshape(1, 1, 1).float()
        
        # Update input sequence
        input_seq = torch.cat((input_seq[:, 1:, :], next_step), dim=1)
    
    # Convert back to original scale
    rolling_preds_np = scaler.inverse_transform(np.array(rolling_preds).reshape(-1, 1))
    return rolling_preds_np.flatten()


def multi_step_forecast(model, X_test, scaler, steps_ahead=5):
    """
    Alternative: Multi-step ahead predictions to reduce error accumulation
    """
    predictions = []
    
    for i in range(0, len(X_test), steps_ahead):
        input_seq = X_test[i].unsqueeze(0)
        
        # Predict multiple steps at once
        step_preds = []
        for step in range(min(steps_ahead, len(X_test) - i)):
            with torch.no_grad():
                pred = model(input_seq)
            step_preds.append(pred.item())
            
            # Update sequence with prediction
            next_step = pred.reshape(1, 1, 1)
            input_seq = torch.cat((input_seq[:, 1:, :], next_step), dim=1)
        
        predictions.extend(step_preds)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def ensemble_rolling_forecast(models, X_test, scaler, weights=None):
    """
    Ensemble approach using multiple models
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    ensemble_preds = []
    input_seqs = [X_test[0].unsqueeze(0).clone() for _ in models]
    
    for i in range(len(X_test)):
        step_preds = []
        
        # Get prediction from each model
        for j, model in enumerate(models):
            with torch.no_grad():
                pred = model(input_seqs[j])
            step_preds.append(pred.item())
            
            # Update input sequence for this model
            next_step = pred.reshape(1, 1, 1)
            input_seqs[j] = torch.cat((input_seqs[j][:, 1:, :], next_step), dim=1)
        
        # Weighted ensemble prediction
        ensemble_pred = sum(w * p for w, p in zip(weights, step_preds))
        ensemble_preds.append(ensemble_pred)
    
    return scaler.inverse_transform(np.array(ensemble_preds).reshape(-1, 1)).flatten()


def adaptive_window_forecast(model, X_test, y_test, scaler, 
                           initial_window=60, min_window=30, max_window=120):
    """
    Adaptive window size based on prediction accuracy
    """
    rolling_preds = []
    window_size = initial_window
    input_seq = X_test[0].unsqueeze(0)
    
    recent_accuracy = []
    
    for i in range(len(X_test)):
        with torch.no_grad():
            pred = model(input_seq)
        
        rolling_preds.append(pred.item())
        
        # Track accuracy and adjust window size
        if i < len(y_test):
            actual = y_test[i].item()
            accuracy = 1 - abs(pred.item() - actual) / (abs(actual) + 1e-8)
            recent_accuracy.append(accuracy)
            
            if len(recent_accuracy) > 10:
                recent_accuracy.pop(0)
                
                avg_accuracy = np.mean(recent_accuracy)
                if avg_accuracy < 0.8 and window_size < max_window:
                    window_size = min(window_size + 5, max_window)
                elif avg_accuracy > 0.9 and window_size > min_window:
                    window_size = max(window_size - 5, min_window)
        
        # Update input sequence with adaptive window
        next_step = pred.reshape(1, 1, 1)
        if input_seq.shape[1] >= window_size:
            # Trim to desired window size
            trim_amount = input_seq.shape[1] - window_size + 1
            input_seq = torch.cat((input_seq[:, trim_amount:, :], next_step), dim=1)
        else:
            input_seq = torch.cat((input_seq[:, 1:, :], next_step), dim=1)
    
    return scaler.inverse_transform(np.array(rolling_preds).reshape(-1, 1)).flatten()

