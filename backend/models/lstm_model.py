import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of hidden units in the LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of output units (usually 1 for regression).
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTM model. This method defines the computation performed at every cell.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor: Predicted output of shape (batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # initial cell state

        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out