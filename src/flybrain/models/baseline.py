"""A placeholder baseline policy.

.. warning::
   **This is not a FlyBrain model.** :class:`TinyBaselineNet` is a plain
   fully-connected policy/value network. It exists to prove that the
   observation format, the action space and the training loop are wired
   together correctly, and to act as the control arm that a future
   connectome-initialised network has to beat. It has no relationship to the
   fruit-fly connectome, and its weights are not biologically meaningful.

   The layers here are interchangeable with a spiking/connectome-derived circuit
   of the same input and output dimensionality; only the internals change.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

__all__ = ["BaselineConfig", "TinyBaselineNet", "count_parameters", "resolve_device"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def resolve_device(requested: str = "cpu") -> torch.device:
    """Resolve a device string, falling back to CPU with a warning.

    GPU acceleration is optional everywhere in FlyBrain: a missing CUDA build
    degrades performance, never correctness.
    """
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"device '{requested}' requested but CUDA is not available; falling back to CPU",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


@dataclass
class BaselineConfig:
    """Shape and placement of the baseline network."""

    input_dim: int = 153
    num_actions: int = 3
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.num_actions <= 0:
            raise ValueError("input_dim and num_actions must be positive")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden sizes must be positive")
        self.hidden_sizes = tuple(int(size) for size in self.hidden_sizes)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None, input_dim: int, num_actions: int) -> "BaselineConfig":
        """Build a config from the ``model.baseline`` YAML section."""
        if not mapping:
            return cls(input_dim=input_dim, num_actions=num_actions)
        known = {"hidden_sizes", "activation", "device", "seed"}
        kwargs: dict[str, Any] = {str(k): v for k, v in mapping.items() if str(k) in known}
        if "hidden_sizes" in kwargs:
            kwargs["hidden_sizes"] = tuple(int(size) for size in kwargs["hidden_sizes"])
        return cls(input_dim=input_dim, num_actions=num_actions, **kwargs)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters, for fair comparison across models."""
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


class TinyBaselineNet(nn.Module):
    """A small MLP with a policy head and a value head.

    Example
    -------
    >>> import numpy as np
    >>> net = TinyBaselineNet(BaselineConfig(input_dim=8, num_actions=3))
    >>> obs = np.zeros(8, dtype=np.float32)
    >>> action, log_prob, value = net.act(obs)
    >>> 0 <= int(action) < 3
    True
    """

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__()
        self.config = config or BaselineConfig()
        torch.manual_seed(self.config.seed)

        activation = _ACTIVATIONS[self.config.activation]
        layers: list[nn.Module] = []
        in_features = self.config.input_dim
        for size in self.config.hidden_sizes:
            layers += [nn.Linear(in_features, size), activation()]
            in_features = size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_features, self.config.num_actions)
        self.value_head = nn.Linear(in_features, 1)

        self.to(self.device)

    # ---------------------------------------------------------------- device

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # --------------------------------------------------------------- forward

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, value)`` for a batch of observations."""
        features = self.trunk(self.as_tensor(observation))
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def distribution(self, observation: torch.Tensor) -> Categorical:
        """Categorical policy over actions."""
        logits, _ = self.forward(observation)
        return Categorical(logits=logits)

    def act(
        self,
        observation: np.ndarray | torch.Tensor,
        greedy: bool = False,
    ) -> tuple[int, float, float]:
        """Sample an action.

        Returns
        -------
        tuple[int, float, float]
            ``(action, log_prob, value)`` for a single observation. Scalars are
            detached Python floats so callers can log them directly.
        """
        tensor = self.as_tensor(observation)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        logits, value = self.forward(tensor)
        distribution = Categorical(logits=logits)
        action_tensor = torch.argmax(logits, dim=-1) if greedy else distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        return int(action_tensor.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, entropies, values)`` for a batch, for training."""
        logits, value = self.forward(observation)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), value

    # ---------------------------------------------------------------- tensor

    def as_tensor(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Coerce an observation to a batched float tensor on the model device."""
        tensor = observation if isinstance(observation, torch.Tensor) else torch.as_tensor(np.asarray(observation))
        tensor = tensor.to(dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Save weights plus the config needed to rebuild the model."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, target)
        return target

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TinyBaselineNet":
        """Rebuild a model saved with :meth:`save`."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(BaselineConfig(**payload["config"])).to(resolve_device(device))

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"input_dim={self.config.input_dim}, num_actions={self.config.num_actions}"
