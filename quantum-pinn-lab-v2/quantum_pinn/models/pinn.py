"""Tanh MLP trial wavefunction with exact Dirichlet boundary enforcement."""
from __future__ import annotations

import torch
from torch import nn


class WavefunctionPINN(nn.Module):
    def __init__(self, xmin: float, xmax: float, width: int = 64, hidden_layers: int = 4):
        super().__init__()
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        blocks: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            blocks.extend([nn.Linear(width, width), nn.Tanh()])
        blocks.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*blocks)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (x - self.xmin) / (self.xmax - self.xmin) - 1.0
        # The envelope forces psi(xmin)=psi(xmax)=0 exactly.
        return (1.0 - z**2) * self.network(z)
