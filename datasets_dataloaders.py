import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        """
        Args:
            data (np.array, 1): The input data array.
            input_window (int): The size of the input window.
            output_window (int): The size of the output window.
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.is1D = len(data.shape) == 1

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        if self.is1D:
            x = self.data[idx : idx + self.input_window]
            y = self.data[idx + self.input_window : idx + self.input_window + self.output_window]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            x = self.data[idx : idx + self.input_window, 0]
            y = self.data[idx + self.input_window : idx + self.input_window + self.output_window, 0]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

class MultiTimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        """
        Args:
            data (np.array): The input data array with shape (time_steps, num_series).
            input_window (int): The size of the input window.
            output_window (int): The size of the output window.
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.num_series = data.shape[1]

    def __len__(self):
        return (len(self.data) - self.input_window - self.output_window + 1) * self.num_series

    def __getitem__(self, idx):
        # Calculate which series and time step this index corresponds to
        series_idx = idx // (len(self.data) - self.input_window - self.output_window + 1)
        time_idx = idx % (len(self.data) - self.input_window - self.output_window + 1)
        
        # Extract the input and output sequences for the corresponding series and time step
        x = self.data[time_idx:time_idx+self.input_window, series_idx]
        y = self.data[time_idx+self.input_window:time_idx+self.input_window+self.output_window, series_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage for a single TS
    data = np.random.randn(8760,1)  # Replace this with your actual DataFrame data

    input_window = 24  # Input window size (e.g., past 24 hours)
    output_window = 12  # Output window size (e.g., next 12 hours)

    dataset = TimeSeriesDataset(data, input_window, output_window)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example of iterating through the dataloader
    print(len(dataloader))
    for batch_x, batch_y in dataloader:
        print(batch_x.shape, batch_y.shape)  # Should be (batch_size, input_window) and (batch_size, output_window)
        break
    
    
    # Example usage for multi TS
    data = np.random.randn(8760, 2000)  # Replace this with your actual DataFrame data

    dataset = MultiTimeSeriesDataset(data, input_window, output_window)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example of iterating through the dataloader
    print(len(dataloader))
    for batch_x, batch_y in dataloader:
        print(batch_x.shape, batch_y.shape)  # Should be (batch_size, input_window) and (batch_size, output_window)
        break