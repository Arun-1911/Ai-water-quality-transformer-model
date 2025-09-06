import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load Dataset
df = pd.read_csv("water_potability.csv")

# Fill missing values with column means
df.fillna(df.mean(), inplace=True)

# Split features & labels
X = df.drop(columns=["Potability"]).values  # Features
y = df["Potability"].values  # Labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))  # FIXED DIMENSION

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.3),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding  # Apply Positional Encoding
        x = self.transformer(x.unsqueeze(1))  # Add sequence dimension
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)

# Model, Loss & Optimizer
input_dim = X_train.shape[1]
num_classes = 2  # Potability (0 or 1)
model = TransformerModel(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training Loop
def train_model(model, train_loader, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Run Training
train_model(model, train_loader, epochs=10)
