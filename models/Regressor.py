from torch import nn

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_sequence = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.linear_sequence(x)