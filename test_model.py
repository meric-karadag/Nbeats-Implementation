import torch
from model import NBEATS

# Example usage:
input_size = 72  # Example input size (e.g., 72 time steps)
hidden_size = 128  # Example hidden size
output_size = 12  # Example output size (e.g., 12 time steps forecast)
num_blocks = 5  # Example number of blocks
num_layers = 4  # Example number of layers per block

model = NBEATS(input_size, hidden_size, output_size, num_layers, num_blocks)
x = torch.randn(32, input_size)  # Example batch of input data (batch size 32)
forecast = model(x)
print(forecast.shape)  # Should output torch.Size([32, output_size])
