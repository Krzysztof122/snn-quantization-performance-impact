from torch import nn


class MLPsmall(nn.Module):
    def __init__(self, actv="relu"):
        super().__init__()

        match actv.lower():
            case "sigmoid":
                activation = nn.Sigmoid()
            case "tanh":
                activation = nn.Tanh()
            case _:
                activation = nn.ReLU()

        self.linear_stack = nn.Sequential(
            nn.Linear(13,16),
            activation,
            nn.Linear(16,3),
        )
        
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


