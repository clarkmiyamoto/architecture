import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_widths):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28 * 28, layer_widths[0]))
        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i+1]))
        self.layers.append(nn.Linear(layer_widths[-1], 10))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # No ReLU on the output layer
        return x