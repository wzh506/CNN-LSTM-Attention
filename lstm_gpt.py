from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


import pandas as pd

# Load the Excel file
file_path = '/mnt/data/merged_data.xlsx'

# Read the data
data = pd.read_excel(file_path)

# Show the first few rows of the dataset to understand its structure
data.head()


# Extract relevant columns for input and target variables
input_columns = ['lrad', 'prec', 'srad', 'Tmax', 'Tmin', 'wind', 'ET0', 'SPEI']
target_columns = ['Yield', 'Trend Yield', 'Relative Meteorological Yield']

# Filter the data
data_filtered = data[input_columns + target_columns]

# Fill missing values with column mean (if any)
data_filtered = data_filtered.fillna(data_filtered.mean())

# Standardize the input features
scaler = StandardScaler()
scaled_inputs = scaler.fit_transform(data_filtered[input_columns])

# Use the last 3 columns as targets
targets = data_filtered[target_columns].values

# Reshape the data for LSTM: 3D array (samples, timesteps, features)
# For simplicity, we'll use all features as a single timestep
X = scaled_inputs
y = targets

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM expects input of shape (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # Add the sequence length dimension
        lstm_out, _ = self.lstm(x)
        # Use the output of the last timestep
        output = self.fc(lstm_out[:, -1, :])
        return output

# Initialize and train the LSTM model
input_size = X_train.shape[1]  # Number of features
hidden_size = 64  # Number of LSTM hidden units
output_size = len(target_columns)  # Number of target variables (3: Yield, Trend Yield, Relative Meteorological Yield)

model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print average loss per epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "/mnt/data/lstm_model.pth")  # Save the model to a file

# Example: Test the model on the test set (optional)
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")
