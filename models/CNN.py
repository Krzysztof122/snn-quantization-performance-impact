from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8*8*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


