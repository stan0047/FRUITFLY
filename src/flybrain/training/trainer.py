"""REINFORCE with a learned value baseline.

Minimal on purpose: enough to demonstrate that observations, actions, rewards and
the policy connect correctly, and nothing more. Any behaviour claimed about the
fly brain would be meaningless at this stage, so the algorithm is deliberately
the simplest standard one.

This module knows about environments, rewards and gradient steps. It does **not**
know about connectomes, membrane potentials or plots, which keeps the
simulation/RL separation intact.

TODO(next milestone): replace REINFORCE with PPO. The sample-reward baseline used
here has high variance, which makes results noisy enough to be hard to compare
across seeds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from flybrain.models.baseline import TinyBaselineNet, resolve_device

__all__ = ["TrainReport", "Trainer", "TrainerConfig"]

ProgressCallback = Callable[[int, int, "TrainReport"], None]


@dataclass
class TrainerConfig:
    """Optimisation and bookkeeping settings."""

    episodes: int = 20
    total_timesteps: int = 2000
    learning_rate: float = 0.001
    gamma: float = 0.99
    grad_clip: float = 1.0
    log_interval: int = 1
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TrainerConfig":
        if not mapping:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{str(k): v for k, v in mapping.items() if str(k) in known})


@dataclass
class TrainReport:
    """Outcome of one :meth:`Trainer.train` call."""

    episodes: int = 0
    timesteps: int = 0
    successes: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_length: float = 0.0
    wall_time: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    note: str = "Baseline policy trained with REINFORCE. Not a fly-brain model; no biological claim."

    def finalize(self) -> "TrainReport":
        """Compute the aggregate statistics from the recorded per-episode data."""
        self.episodes = len(self.episode_rewards)
        self.mean_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        self.mean_length = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        self.success_rate = self.successes / max(self.episodes, 1)
        return self

    def summary(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "timesteps": self.timesteps,
            "mean_reward": self.mean_reward,
            "mean_length": self.mean_length,
            "success_rate": self.success_rate,
            "final_losses": self.losses[-3:],
            "wall_time": self.wall_time,
            "note": self.note,
        }


class Trainer:
    """Run episodes, compute discounted returns, and take one gradient step per episode."""

    def __init__(
        self,
        model: TinyBaselineNet,
        config: TrainerConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.report = TrainReport(config=asdict(self.config))

    # ------------------------------------------------------------------ train

    def train(
        self,
        env: Any,
        num_episodes: int | None = None,
        total_timesteps: int | None = None,
        callback: ProgressCallback | None = None,
    ) -> TrainReport:
        """Train for a number of episodes, stopping early at ``total_timesteps``.

        Parameters
        ----------
        env:
            Any object implementing the gymnasium ``reset``/``step`` contract.
        callback:
            Optional ``callback(timesteps, total_timesteps, report)`` hook, used
            for progress bars without adding a dependency.
        """
        episodes = int(num_episodes or self.config.episodes)
        timestep_budget = int(total_timesteps or self.config.total_timesteps)
        report = TrainReport(config=asdict(self.config))
        started = time.perf_counter()

        for episode in range(episodes):
            if report.timesteps >= timestep_budget:
                break

            observations, actions, rewards = self._collect_episode(env, report, timestep_budget)
            if actions.numel() == 0:
                break

            loss = self._optimise(observations, actions, rewards)
            loss_value = float(loss.detach())
            report.losses.append(loss_value)

            if self.config.verbose and self.config.log_interval and (episode + 1) % self.config.log_interval == 0:
                print(
                    f"episode {episode + 1:4d}/{episodes}  "
                    f"timesteps={report.timesteps:6d}  "
                    f"reward={report.episode_rewards[-1]:8.3f}  "
                    f"length={report.episode_lengths[-1]:3d}  "
                    f"loss={loss_value:+.5f}"
                )
            if callback is not None:
                callback(report.timesteps, timestep_budget, report)
            self._maybe_checkpoint(episode)

        report.wall_time = time.perf_counter() - started
        report.finalize()
        self.report = report
        return report

    def _collect_episode(
        self,
        env: Any,
        report: TrainReport,
        timestep_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Run one episode, filling the report as we go.

        Returns the stacked observations, the actions taken and the raw rewards.
        Log-probabilities are recomputed inside :meth:`_optimise` from the
        current parameters, which is what a policy-gradient update needs.
        """
        observation, _ = env.reset()
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        done = False
        while not done and report.timesteps < timestep_budget:
            action, _, _ = self.model.act(observation)
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(action)
            rewards.append(0.0)

            observation, reward, terminated, truncated, info = env.step(action)
            rewards[-1] = float(reward)
            report.timesteps += 1
            done = bool(terminated) or bool(truncated)

        report.episode_rewards.append(float(np.sum(rewards)))
        report.episode_lengths.append(len(rewards))
        if isinstance(info, Mapping) and info.get("success", False):
            report.successes += 1

        obs_tensor = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=self.device)
        return obs_tensor, action_tensor, rewards

    def _optimise(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: list[float],
    ) -> torch.Tensor:
        """One REINFORCE update using the value head as the baseline."""
        returns = self._discounted_returns(rewards)
        advantage = returns - self.model.value_head(self.model.trunk(observations)).squeeze(-1)
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        new_log_probs, entropy, values = self.model.evaluate_actions(observations, actions)
        value_loss = nn.functional.mse_loss(values, returns)
        policy_loss = -(new_log_probs * advantage.detach()).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        """Reverse-time discounted sum, returned in forward order."""
        returns = torch.zeros(len(rewards), dtype=torch.float32, device=self.device)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = float(rewards[index]) + self.config.gamma * running
            returns[index] = running
        return returns

    # ------------------------------------------------------------ evaluation

    def evaluate(self, env: Any, episodes: int = 5, greedy: bool = True) -> dict[str, float]:
        """Roll out without gradients and report mean reward and success rate."""
        rewards: list[float] = []
        lengths: list[int] = []
        successes = 0

        for _ in range(max(episodes, 0)):
            observation, _ = env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done:
                action, _, _ = self.model.act(observation, greedy=greedy)
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
                done = bool(terminated) or bool(truncated)
            rewards.append(total)
            lengths.append(steps)
            successes += int(bool(info.get("success", False))) if isinstance(info, Mapping) else 0

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "success_rate": successes / max(len(rewards), 1),
            "episodes": float(len(rewards)),
        }

    # ------------------------------------------------------------ checkpoints

    def _maybe_checkpoint(self, episode: int) -> None:
        every = self.config.checkpoint_every
        if not every or not self.config.checkpoint_dir:
            return
        if (episode + 1) % every == 0:
            self.save_checkpoint(Path(self.config.checkpoint_dir) / f"checkpoint_ep{episode + 1:04d}.pt")

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist model weights, optimiser state and the training report."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "report": self.report.summary(),
            },
            target,
        )
        return target
