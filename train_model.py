import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import NBEATS
from datasets_dataloaders import MultiTimeSeriesDataset
from metrics import my_R2Score
import numpy as np
import wandb
import pandas as pd
import random

# Initialize WandB
wandb.init(project='Train N-BEATS')

# Hyperparameters
input_window = 72
output_window = 6
hidden_size = 128
num_layers = 4
num_blocks = 15
batch_size = 512
learning_rate = 5e-4
num_epochs = 10
early_stopping_patience = 5
torch.manual_seed(42)

# Load the data
city = "Ankara"
data = pd.read_csv(f"./{city}/{city}_Hourly_W_Date.csv")
random_cols = [str(random.randint(2000, 3999))+'_0' for _ in range(200)]
data = data[random_cols]
data = data.to_numpy(dtype=np.float32)

# Create the dataset
dataset = MultiTimeSeriesDataset(data, input_window, output_window)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NBEATS(input_window, hidden_size, output_window,num_layers, num_blocks)
model.to(device)

# Initialize the R² score metric
r2Score = my_R2Score(num_outputs = 6)
r2Score.to(device)

r2Score0 = my_R2Score(position=0)
r2Score0.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping parameters
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    running_train_r2 = 0.0
    running_train_r2_0 = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_x, batch_y in progress_bar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        wandb.log({'Loss/train_batch': loss.item()})
        running_train_loss += loss.item() * batch_x.size(0)
        running_train_r2 += r2Score(outputs, batch_y) * batch_x.size(0)
        running_train_r2_0 += r2Score0(outputs, batch_y) * batch_x.size(0)
        
        progress_bar.set_postfix(loss=running_train_loss / len(train_dataloader.dataset), r2=running_train_r2 / len(train_dataloader.dataset))
    
    epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
    epoch_train_r2 = running_train_r2 / len(train_dataloader.dataset)
    epoch_train_r2_0 = running_train_r2_0 / len(train_dataloader.dataset)
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    running_val_r2 = 0.0
    running_val_r2_0 = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            running_val_loss += loss.item() * batch_x.size(0)
            running_val_r2 += r2Score(outputs, batch_y) * batch_x.size(0)
            running_val_r2_0 += r2Score0(outputs, batch_y) * batch_x.size(0)

    epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
    epoch_val_r2 = running_val_r2 / len(val_dataloader.dataset)
    epoch_val_r2_0 = running_val_r2_0 / len(val_dataloader.dataset)    
    # Log metrics to WandB
    wandb.log({
        'Loss/train': epoch_train_loss,
        'R2/train': epoch_train_r2,
        'R2/train_0': epoch_train_r2_0,
        'Loss/val': epoch_val_loss,
        'R2/val': epoch_val_r2,
        'R2/val': epoch_val_r2_0
    }, step=epoch)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train R²: {epoch_train_r2:.4f}, Val Loss: {epoch_val_loss:.4f}, Val R²: {epoch_val_r2:.4f}")
    
    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

print("Training complete")