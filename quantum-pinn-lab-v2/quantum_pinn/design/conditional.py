"""Parameter-conditioned multi-state wavefunction and energy surrogate."""
from __future__ import annotations

import torch
from torch import nn


class ParameterConditionedSurrogate(nn.Module):
    """Predict psi_n(x; p) and E_n(p) from position, potential parameters and state index.

    The coordinate network uses a hard Dirichlet envelope, while the energy head is
    coordinate-independent by construction.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n_parameters: int,
        n_states: int = 2,
        width: int = 96,
        hidden_layers: int = 4,
        family: str | None = None,
    ) -> None:
        super().__init__()
        if n_parameters < 1 or n_states < 1 or width < 8 or hidden_layers < 1:
            raise ValueError("Invalid conditional model dimensions")
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.n_parameters = int(n_parameters)
        self.n_states = int(n_states)
        self.width = int(width)
        self.hidden_layers = int(hidden_layers)
        self.family = family
        self.wavefunction_net = self._mlp(1 + n_parameters + 1, 1)
        self.energy_net = self._mlp(n_parameters + 1, 1)

    def _mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, self.width), nn.Tanh()]
        for _ in range(self.hidden_layers - 1):
            layers.extend([nn.Linear(self.width, self.width), nn.Tanh()])
        layers.append(nn.Linear(self.width, output_dim))
        network = nn.Sequential(*layers)
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        return network

    def _state_feature(self, state_index: torch.Tensor | int, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(state_index, int):
            indices = torch.full((batch, 1), float(state_index), device=device, dtype=dtype)
        else:
            indices = state_index.to(device=device, dtype=dtype).reshape(-1, 1)
            if indices.shape[0] == 1 and batch > 1:
                indices = indices.expand(batch, 1)
        if indices.shape[0] != batch:
            raise ValueError("state_index batch dimension does not match x")
        if torch.any(indices < 0) or torch.any(indices >= self.n_states):
            raise ValueError("state_index lies outside model's trained state range")
        return indices / max(self.n_states - 1, 1)

    def forward(
        self, x: torch.Tensor, normalized_parameters: torch.Tensor, state_index: torch.Tensor | int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if normalized_parameters.ndim == 1:
            normalized_parameters = normalized_parameters.reshape(1, -1)
        if normalized_parameters.shape[1] != self.n_parameters:
            raise ValueError(f"Expected {self.n_parameters} normalized parameters")
        if normalized_parameters.shape[0] == 1 and x.shape[0] > 1:
            normalized_parameters = normalized_parameters.expand(x.shape[0], -1)
        if normalized_parameters.shape[0] != x.shape[0]:
            raise ValueError("Parameter batch dimension does not match x")
        state = self._state_feature(state_index, x.shape[0], x.device, x.dtype)
        z = 2.0 * (x - self.xmin) / (self.xmax - self.xmin) - 1.0
        psi_features = torch.cat((z, normalized_parameters, state), dim=1)
        energy_features = torch.cat((normalized_parameters, state), dim=1)
        psi = (1.0 - z**2) * self.wavefunction_net(psi_features)
        energy = self.energy_net(energy_features).squeeze(-1)
        return psi.squeeze(-1), energy
