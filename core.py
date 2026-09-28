# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted and simplified from ms-swift; modifications documented in NOTICE.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Config:
    group_size: int = 4
    beta: float = 0.04
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    scale_rewards: str = "group"
    loss_type: str = "grpo"
    max_completion_length: int = 256
    updates_per_rollout: int = 1
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if self.group_size < 2 or self.updates_per_rollout < 1:
            raise ValueError("group_size >= 2 and updates_per_rollout >= 1 required")
        if self.beta < 0 or not 0 <= self.epsilon_low < 1 or self.epsilon_high < 0:
            raise ValueError("Invalid KL/clipping coefficients")
        if self.scale_rewards not in {"group", "batch", "none"}:
            raise ValueError("Unsupported reward scaling")
        if self.loss_type not in {"grpo", "bnpo", "dr_grpo"}:
            raise ValueError("Unsupported loss type")
        if self.max_completion_length < 1 or self.max_grad_norm <= 0:
            raise ValueError("Invalid length/gradient limit")


def advantages(rewards_per_func, weights, config):
    """Rows must be contiguous groups: x0/g0, x0/g1, ..., x1/g0.

    NaN means this reward is inapplicable. All-missing rows are rejected.
    std uses Bessel correction, matching the pinned ms-swift implementation.
    """
    r = rewards_per_func.detach().float()
    w = torch.as_tensor(weights, dtype=r.dtype, device=r.device)
    if r.ndim != 2 or r.shape[0] == 0 or r.shape[0] % config.group_size:
        raise ValueError("Expected nonempty [B*G, reward_functions] matrix")
    if w.shape != (r.shape[1],) or not torch.isfinite(w).all() or not (w != 0).any():
        raise ValueError("Invalid reward weights")
    if torch.isinf(r).any() or torch.isnan(r[:, w != 0]).all(dim=1).any():
        raise ValueError("Every sample needs at least one finite active reward")
    total = (r * w).nansum(dim=1)
    groups = total.view(-1, config.group_size)
    adv = total - groups.mean(1).repeat_interleave(config.group_size)
    if config.scale_rewards == "group":
        adv = adv / (groups.std(1).repeat_interleave(config.group_size) + 1e-4)
    elif config.scale_rewards == "batch":
        adv = adv / (total.std() + 1e-4)
    return adv, total


def completion_mask(token_ids, lengths, eos_ids):
    """Completion-only IDs [N,T]; lengths exclude padding, include emitted EOS.

    Explicit lengths disambiguate EOS from PAD even when they share an ID.
    Empty transcription still contains one EOS token and can be optimized.
    Returns (mask including first EOS, truncated/no-EOS flags).
    """
    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError("Expected nonempty [N,T] completion IDs")
    lengths = torch.as_tensor(lengths, device=token_ids.device)
    if lengths.shape != (token_ids.shape[0],) or (lengths < 1).any() or (lengths > token_ids.shape[1]).any():
        raise ValueError("Invalid generated lengths")
    positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
    valid = positions < lengths[:, None]
    eos = torch.zeros_like(valid)
    for token in eos_ids:
        eos |= token_ids.eq(token) & valid
    first_eos = torch.where(eos, positions, token_ids.shape[1]).amin(1)
    return valid & (positions <= first_eos[:, None]), ~eos.any(1)


def grpo_loss(logps, old_logps, ref_logps, mask, adv, config):
    """Token-level clipped GRPO; logps are completion-only, causally aligned.

    Matches the selected ms-swift branch including its bounded k3 estimator.
    No distributed/gradient-accumulation normalization is implied here.
    """
    if logps.ndim != 2 or old_logps.shape != logps.shape or mask.shape != logps.shape:
        raise ValueError("logps, old_logps, mask must share [N,T] shape")
    if mask.dtype != torch.bool or adv.shape != (logps.shape[0],):
        raise ValueError("Expected bool mask and sequence-level advantages")
    if not mask.any(1).all():
        raise ValueError("Every completion must contain a trainable token (including EOS)")
    if config.beta and (ref_logps is None or ref_logps.shape != logps.shape):
        raise ValueError("Reference log probabilities required when beta > 0")
    if not torch.isfinite(adv).all():
        raise ValueError("Nonfinite advantages")
    current = logps.float().masked_fill(~mask, 0)
    old = old_logps.detach().float().masked_fill(~mask, 0)
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("Nonfinite active log probabilities")
    ratio = (current - old).exp()
    clipped = ratio.clamp(1 - config.epsilon_low, 1 + config.epsilon_high)
    a = adv.detach().float()[:, None]
    per_token = -torch.minimum(ratio * a, clipped * a)
    kl = torch.zeros_like(current)
    if config.beta:
        reference = ref_logps.detach().float().masked_fill(~mask, 0)
        if not torch.isfinite(reference).all():
            raise ValueError("Nonfinite reference log probabilities")
        delta = (reference - current).clamp(-20, 20)
        kl = (delta.exp() - delta - 1).clamp(-10, 10)
        per_token = per_token + config.beta * kl
    masked = per_token.masked_fill(~mask, 0)
    if config.loss_type == "grpo":
        loss = (masked.sum(1) / mask.sum(1)).mean()
    elif config.loss_type == "bnpo":
        loss = masked.sum() / mask.sum()
    else:
        if (mask.sum(1) > config.max_completion_length).any():
            raise ValueError("Completion exceeds dr_grpo denominator length")
        loss = masked.sum() / (mask.shape[0] * config.max_completion_length)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite GRPO loss; optimizer step cancelled")
    clip = ((ratio < 1 - config.epsilon_low) & (a < 0)) | ((ratio > 1 + config.epsilon_high) & (a > 0))
    return loss, {
        "loss": loss.detach().item(),
        "kl": kl.masked_select(mask).mean().detach().item(),
        "clip_fraction": clip.masked_select(mask).float().mean().item(),
    }
