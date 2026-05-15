import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from optionlib.ml.data_gen import generate_data
from optionlib.ml.model import OptionPricer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler          # Normalization
import joblib                                           # to save data

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train(n_samples = 100000, epochs = 50, lr = 0.001, batch_size = 256):
    """
    Training the model
    """
    
    df = generate_data(n_samples=n_samples)             # Generating data
    features = df.drop(columns=["price"]).values        # Features df
    target = df["price"].values.reshape(-1,1)           # target = price

    scaler = MinMaxScaler()                             # normalize features to same scale
    features = scaler.fit_transform(features)

    features_train, features_val, target_train, target_val =train_test_split(features, target, test_size=0.2, random_state=42)          # 80/20 train/validation split

    joblib.dump(scaler, os.path.join(BASE_DIR, "training", "scaler.pkl"))

    features_train = torch.FloatTensor(features_train)
    features_val = torch.FloatTensor(features_val)
    target_train = torch.FloatTensor(target_train)
    target_val = torch.FloatTensor(target_val)                  # Converting everything to Float Tensor

    # Model
    model = OptionPricer()
    optimizer = optim.Adam(model.parameters(), lr=lr)           # Adam Optimizer -> Gradient descent
    criterion = nn.MSELoss()                                    # Loss function

    # Training loop
    for epoch in range(epochs):
        idx = torch.randperm(len(features_train))               # Permutation of indeces so the model doesn't memorize the order
        
        total_loss = 0
        
        for batch in range(0, len(features_train), batch_size):
            X_batch = features_train[idx[batch:batch + batch_size]]   # Isolating the features
            Y_batch = target_train[idx[batch:batch + batch_size]]     # Isolating the target

            optimizer.zero_grad()                               # Zeroing the gradients

            predictions = model(X_batch)                        # Running the model
            loss = criterion(predictions, Y_batch)              # Computing the loss
            loss.backward()                                     # Backpropagation
            total_loss += loss.item()                           # Total loss calculation
            optimizer.step()

        model.eval()                                            # switch to eval mode for validation
        with torch.no_grad():
            val_predictions = model(features_val)
            val_loss = criterion(val_predictions, target_val)
        model.train()                                           # switch back to training mode

        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss / (len(features_train) // batch_size):.4f}  |  Val Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "training", "model_weights.pt"))       # save trained weights
    return model