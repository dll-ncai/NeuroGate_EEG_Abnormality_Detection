import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

class ShallowAutoencoder(nn.Module):
    def __init__(self, input_size=50, hidden_size=32):
        super(ShallowAutoencoder, self).__init__()
        self.input_size = input_size
        
        # Encoder: Linear + Satlin (Hardtanh)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Hardtanh(min_val=-1.0, max_val=1.0) 
        )
        
        # Decoder: Linear + Purelin (No activation)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded