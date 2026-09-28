# SPDX-License-Identifier: Apache-2.0
"""Synchronous single-process loop. Audio/model details live in an adapter."""
from dataclasses import dataclass
from typing import Any, Protocol
import torch
from .core import Config, advantages, grpo_loss


@dataclass
class Rollout:
    payload: Any  # Immutable audio inputs + prompt IDs + sampled completion IDs.
    texts: list[str]
    mask: torch.Tensor  # [B*G,T], completion-only, includes EOS.


class Adapter(Protocol):
    def sample(self, policy, batch, group_size: int) -> Rollout:
        """Sample contiguous groups from policy, preserving identical audio per group.

        Must enforce max length; return bool completion masks including EOS.
        The minimal on-policy contract requires temperature=1, no top-k/top-p
        truncation or repetition penalties, matching logps()'s distribution.
        """
        ...

    def logps(self, model, rollout: Rollout) -> torch.Tensor:
        """Return differentiable [B*G,T] logps of sampled tokens.

        Use the same waveform/features/prompt and causal shift as sample().
        Do not cache detached encoder embeddings if encoder is trainable.
        """
        ...


class Trainer:
    def __init__(self, policy, reference, adapter, optimizer, reward_fn, config=Config()):
        if torch.distributed.is_initialized():
            raise ValueError("This extracted loop is single-process only")
        if reference is policy:
            raise ValueError("Reference must be a separate fixed SFT snapshot")
        if config.beta and reference is None:
            raise ValueError("beta > 0 requires a reference model")
        if reference is not None:
            policy_ptrs = {p.data_ptr() for p in policy.parameters()}
            if any(p.data_ptr() in policy_ptrs for p in reference.parameters()):
                raise ValueError("Reference and policy must not share parameters")
            reference.requires_grad_(False)
            reference.eval()
        trainable = {id(p) for p in policy.parameters() if p.requires_grad}
        optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if not trainable or trainable != optimized:
            raise ValueError("Optimizer must contain exactly the trainable policy parameters")
        self.policy, self.reference, self.adapter = policy, reference, adapter
        self.optimizer, self.reward_fn, self.config = optimizer, reward_fn, config

    def step(self, batch):
        # eval disables stochastic dropout, but DOES NOT disable gradients.
        previous_mode = self.policy.training
        self.policy.eval()
        try:
            return self._step(batch)
        finally:
            self.policy.train(previous_mode)

    def _step(self, batch):
        c = self.config
        with torch.no_grad():
            rollout = self.adapter.sample(self.policy, batch, c.group_size)
            old = self.adapter.logps(self.policy, rollout).detach().clone()
            reference = self.adapter.logps(self.reference, rollout).detach() if c.beta else None
            # Callback returns ([B*G,R] scores, [R] weights).
            rewards, weights = self.reward_fn(batch, rollout.texts)
            rewards = torch.as_tensor(rewards, device=old.device, dtype=torch.float32)
            adv, total = advantages(rewards, weights, c)
            if len(rollout.texts) != old.shape[0] or adv.shape[0] != old.shape[0]:
                raise ValueError("Reward/completion count mismatch")
        for _ in range(c.updates_per_rollout):
            self.optimizer.zero_grad(set_to_none=True)
            current = self.adapter.logps(self.policy, rollout)
            loss, metrics = grpo_loss(current, old, reference, rollout.mask, adv, c)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                c.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
        metrics.update(reward=total.mean().item(), grad_norm=grad_norm.item(),
                       zero_std_fraction=(total.view(-1, c.group_size).std(1) == 0).float().mean().item())
        return metrics
