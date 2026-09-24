# arch/mlp.py
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Feed-forward dense MLP with He (Kaiming) initialization.

    This version is intended for **direction regression**:
      - out_dim should be 3
      - forward() returns raw logits (no degree squashing)
      - training script normalizes to unit vectors in loss/metrics

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dims : Sequence[int]
        Hidden layer widths, e.g. [128, 64, 64].
    out_dim : int
        Output feature dimension (use 3 for direction vector).
    activation : str
        One of: {"relu", "leaky_relu", "gelu", "tanh", "sigmoid"}.
    dropout : float
        Dropout probability applied after activation in hidden layers.
    use_batchnorm : bool
        If True, applies BatchNorm1d after each linear (hidden layers only).
    out_activation : Optional[str]
        Optional final activation (usually None).
    leaky_slope : float
        Negative slope for LeakyReLU (if used).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        out_activation: Optional[str] = None,
        leaky_slope: float = 0.01,
    ):
        super().__init__()

        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive integers.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError("All hidden_dims must be positive integers.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        self.in_dim = int(in_dim)
        self.hidden_dims = list(map(int, hidden_dims))
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.activation_name = activation
        self.out_activation_name = out_activation
        self.leaky_slope = float(leaky_slope)

        act = self._make_activation(activation, leaky_slope)
        out_act = self._make_activation(out_activation, leaky_slope) if out_activation else None

        layers: List[nn.Module] = []
        prev = self.in_dim

        # Hidden stack
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h, bias=True))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            prev = h

        # Output layer (raw)
        layers.append(nn.Linear(prev, self.out_dim, bias=True))
        if out_act is not None:
            layers.append(out_act)

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    @staticmethod
    def _make_activation(name: Optional[str], leaky_slope: float) -> nn.Module:
        if name is None:
            raise ValueError("Activation name cannot be None here.")
        name_l = name.lower()
        if name_l == "relu":
            return nn.ReLU(inplace=True)
        if name_l == "leaky_relu":
            return nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        if name_l == "gelu":
            return nn.GELU()
        if name_l == "tanh":
            return nn.Tanh()
        if name_l == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unknown activation: {name}")

    def reset_parameters(self) -> None:
        """
        Initialize Linear weights and biases, and BatchNorm params.
        """
        act = self.activation_name.lower()
        if act == "leaky_relu":
            nonlinearity = "leaky_relu"
            a = self.leaky_slope
        elif act in {"tanh", "sigmoid"}:
            nonlinearity = None  # we'll use Xavier below
            a = 0.0
        else:
            nonlinearity = "relu"
            a = 0.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if nonlinearity is None:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, a=a, mode="fan_in", nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, in_dim)
        returns: (B, out_dim) raw outputs (no angle squashing)
        """
        return self.net(x)
