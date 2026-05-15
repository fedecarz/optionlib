import torch
import torch.nn as nn

class OptionPricer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(       # Sequential Neural Network
            nn.Linear(7, 64),               # input layer: 7 features → 64 neurons
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),               # output layer: 64 neurons → 1 price
        )


    def forward(self, x):
        return self.network(x)
