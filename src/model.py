import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=3):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)