"""Continuous source-audio conditioning for the acoustic decoder."""
from torch import nn
from torch.nn import functional as F


class AudioConditioner(nn.Module):
    def __init__(self, hidden_size, layers, operations):
        super().__init__()
        self.layers = tuple(layers)
        self.projections = nn.ModuleDict({str(layer): operations.Linear(64, hidden_size) for layer in self.layers})

    def forward(self, source):
        source = F.pad(source, (0, 0, 1, 1))[None]
        return {layer: self.projections[str(layer)](source) for layer in self.layers}
