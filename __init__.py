"""Minimal, single-process audio GRPO kernel. See README for integration limits."""
from .core import Config, advantages, completion_mask, grpo_loss
from .trainer import Rollout, Trainer

__all__ = ["Config", "advantages", "completion_mask", "grpo_loss", "Rollout", "Trainer"]
