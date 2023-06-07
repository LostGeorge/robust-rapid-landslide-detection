import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Callable

class MLPDiscriminator(nn.Module):
    def __init__(
            self,
            encoder_dim: int,
            hidden_dims: List[int],
            n_domains: int = 3,
            act_fn: Callable = nn.GELU,
            ) -> None:
        super().__init__()
        assert len(hidden_dims) >= 1
        layers = [
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            nn.Linear(encoder_dim, hidden_dims[0]),
            act_fn(),
        ]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(act_fn())
        layers.append(nn.Linear(hidden_dims[-1], n_domains))
        self.model = nn.Sequential(*layers)

    def forward(self, enc_out):
        return self.model(enc_out)

class LinearDiscriminator(nn.Module):
    def __init__(
            self,
            encoder_dim: int,
            n_domains: int = 3,
            ) -> None:
        super().__init__()
        layers = [
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            nn.Linear(encoder_dim, n_domains),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, enc_out):
        return self.model(enc_out)



