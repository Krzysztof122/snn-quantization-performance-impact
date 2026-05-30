from torch import nn


class MLPbig(nn.Module):
    def __init__(self, actv="relu"):
        super().__init__()
        self.flatten = nn.Flatten()
        match actv.lower():
            case "sigmoid":
                activation = nn.Sigmoid()
            case "tanh":
                activation = nn.Tanh()
            case _:
                activation = nn.ReLU()

        self.linear_stack = nn.Sequential(
            nn.Linear(28*28,512),
            activation,
            nn.Linear(512,512),
            activation,
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


