import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("water_potability.csv")

df.fillna(df.mean(), inplace=True)

# Split features & labels
X = df.drop(columns=["Potability"]).values  # Features
y = df["Potability"].values  # Labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Transformer Model with Positional Encoding
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4, dropout=0.4):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout),
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)  # Removed Softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x += self.positional_encoding  # Add positional encoding
        x = self.transformer(self.norm(x))
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)

# Model Initialization
model = TransformerModel(input_dim=X.shape[1], num_classes=2)

# Class Imbalance Handling
class_counts = np.bincount(y_train.numpy())
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer with L2 Regularization
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)

# Training Function
def train_model(model, train_loader, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Train & Evaluate
train_model(model, train_loader, epochs=10)
evaluate_model(model, test_loader)
