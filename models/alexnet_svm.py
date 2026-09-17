import torch
import torch.nn as nn

class PyTorchRBFSVM(nn.Module):
    def __init__(self, in_features, num_support_vectors=1000):
        super(PyTorchRBFSVM, self).__init__()
        self.gamma = 1.0 / in_features
        self.support_vectors = nn.Parameter(torch.randn(num_support_vectors, in_features))
        self.dual_coef = nn.Parameter(torch.randn(1, num_support_vectors))
        self.intercept = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        dist = torch.cdist(x, self.support_vectors, p=2.0) ** 2
        kernel_vals = torch.exp(-self.gamma * dist)
        return torch.matmul(kernel_vals, self.dual_coef.t()) + self.intercept

class AlexNetSVM(nn.Module):
    def __init__(self, electrodes=22, num_support_vectors=1000):
        super(AlexNetSVM, self).__init__()

        self.features = nn.Sequential(
            # 1. Convolution (10x21, 20) -> Adapted to your 22 electrodes
            nn.Conv2d(1, 20, kernel_size=(electrodes, 10)),
            nn.ReLU(),
            
            # 2. Convolution (10x20, 20) -> Corrected typo: (1, 10)
            nn.Conv2d(20, 20, kernel_size=(1, 10)),
            nn.ReLU(),
            
            # 3. Max-pooling (2x1, stride 2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            # 4. Convolution (10x20, 50)
            nn.Conv2d(20, 50, kernel_size=(1, 10)),
            nn.ReLU(),
            
            # 5. Max-pooling (2x1, stride 2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            # 6. Convolution (10x20, 50)
            nn.Conv2d(50, 50, kernel_size=(1, 10)),
            nn.ReLU(),
            
            # 7. Max-pooling (2x1, stride 2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            # 8. Convolution (10x20, 50)
            nn.Conv2d(50, 50, kernel_size=(1, 10)),
            nn.ReLU(),
            
            # 9. Max-pooling (2x1, stride 2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            # 10. Convolution (10x20, 50)
            nn.Conv2d(50, 50, kernel_size=(1, 10)),
            nn.ReLU(),
            
            # 11. Max-pooling (2x1, stride 2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # Bridge: Forces your 60,000 sequence to match the paper's exact flatten size (50 * 22)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 22))

        self.classifier = nn.Sequential(
            # 12. Fully connected(4096)
            nn.Linear(50 * 22, 4096),
            nn.ReLU(),
            nn.Dropout(0.5), # Paper notes 0.5 dropout on FC layers
            
            # 13. Fully connected(4096)
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # 14. Fully connected(2 classes) -> Replaced by SVM for Transfer Learning
        self.svm = PyTorchRBFSVM(in_features=4096, num_support_vectors=num_support_vectors)

    def forward(self, x):
        # Expected input: [Batch, 1, 22, 60000]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.svm(x)

# --- Profiling Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetSVM().to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 22, 60000 // 2).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape (Batch, 1 decision value): {output.shape}")