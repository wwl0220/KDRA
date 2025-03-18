import torch.nn as nn
import torch.nn.functional as F


class MLP_Model(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.layers = nn.ModuleList()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)

        return x

    def predict_proba(self, x):
        x = F.relu(self.input_layer(x))

        for layer in self.layers:
            x = F.relu(layer(x))

        x = F.softmax(self.output_layer(x), dim=-1)

        return x
