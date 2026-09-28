import copy
import unittest
from functools import partial
import torch
from .core import Config, advantages, completion_mask, grpo_loss
from .rewards import recognition_reward, asr_rewards
from .trainer import Rollout, Trainer


class ToyAdapter:
    """Discrete one-token policy for loop verification, NOT an ASR model."""
    def sample(self, policy, batch, group_size):
        ids = torch.multinomial(policy.weight.softmax(-1), len(batch) * group_size, replacement=True)
        return Rollout(ids, ["好" if i == 0 else "坏" for i in ids],
                       torch.ones(len(ids), 1, dtype=torch.bool))

    def logps(self, model, rollout):
        return model.weight.log_softmax(-1)[rollout.payload, None]


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class Tests(unittest.TestCase):
    def test_advantages(self):
        r = torch.tensor([[1.], [3.], [2.], [2.]])
        a, _ = advantages(r, [1], Config(group_size=2))
        expected = 1 / (2 ** 0.5 + 1e-4)
        torch.testing.assert_close(a, torch.tensor([-expected, expected, 0., 0.]))
        a, _ = advantages(r, [1], Config(group_size=2, scale_rewards="none"))
        torch.testing.assert_close(a, torch.tensor([-1., 1., 0., 0.]))

    def test_missing_rewards(self):
        with self.assertRaises(ValueError):
            advantages(torch.full((2, 1), float("nan")), [1], Config(group_size=2))
        a, r = advantages(torch.tensor([[1., float("nan")], [2., 1.]]),
                          [1., 2.], Config(group_size=2))
        self.assertEqual(r.tolist(), [1., 4.])

    def test_eos_padding_truncation(self):
        mask, truncated = completion_mask(torch.tensor([[0, 0, 0], [2, 0, 0], [1, 2, 3]]),
                                           [1, 2, 3], [0])
        self.assertEqual(mask.tolist(), [[True, False, False], [True, True, False], [True]*3])
        self.assertEqual(truncated.tolist(), [False, False, True])

    def test_clipping_and_gradients(self):
        current = torch.tensor([[0.4, float("nan")], [-0.4, 0.]], requires_grad=True)
        old = torch.zeros_like(current, requires_grad=True)
        mask = torch.tensor([[True, False], [True, False]])
        loss, metrics = grpo_loss(current, old, None, mask, torch.tensor([1., -1.]), Config(beta=0))
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        self.assertEqual(metrics["clip_fraction"], 1.0)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.zeros_like(current))
        self.assertIsNone(old.grad)

    def test_reference_detached_and_kl(self):
        p = torch.tensor([[-1., -2.]], requires_grad=True)
        ref = torch.tensor([[-2., -2.]], requires_grad=True)
        mask = torch.ones_like(p, dtype=torch.bool)
        loss, metrics = grpo_loss(p, p.detach(), ref, mask, torch.zeros(1), Config(beta=0.1))
        self.assertAlmostEqual(metrics["kl"], (torch.exp(torch.tensor(-1.)).item())/2, places=6)
        loss.backward()
        self.assertIsNone(ref.grad)
        self.assertNotEqual(p.grad[0, 0].item(), 0)

    def test_reductions(self):
        p = torch.zeros(2, 3, requires_grad=True)
        mask = torch.tensor([[True, False, False], [True, True, True]])
        for kind, expected in [("grpo", -2.), ("bnpo", -2.5), ("dr_grpo", -10/6)]:
            loss, _ = grpo_loss(p, p.detach(), None, mask, torch.tensor([1., 3.]),
                                Config(beta=0, loss_type=kind, max_completion_length=3))
            self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_asr_reward(self):
        self.assertEqual(recognition_reward("你好。", "你好"), 1.)
        self.assertEqual(recognition_reward("", ""), 1.)
        self.assertEqual(recognition_reward("幻觉", ""), -1.)
        self.assertEqual(recognition_reward("", "语音"), 0.)
        self.assertEqual(recognition_reward("hello WORLD", "hello world", "word"), 1.)

    def test_optimizer_loop(self):
        torch.manual_seed(11)
        policy = ToyPolicy()
        reference = copy.deepcopy(policy)
        frozen = policy.frozen.detach().clone()
        ref_before = copy.deepcopy(reference.state_dict())
        trainer = Trainer(policy, reference, ToyAdapter(),
                          torch.optim.AdamW([policy.weight], lr=0.05),
                          partial(asr_rewards, group_size=16),
                          Config(group_size=16, updates_per_rollout=2))
        for _ in range(8):
            metrics = trainer.step([{"reference": "好"}])
        self.assertGreater(policy.weight[0].item(), policy.weight[1].item())
        self.assertTrue(policy.training)
        torch.testing.assert_close(policy.frozen, frozen)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, ref_before[key])
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))

    def test_reject_shared_reference(self):
        p = ToyPolicy()
        with self.assertRaises(ValueError):
            Trainer(p, p, ToyAdapter(), torch.optim.Adam([p.weight]), asr_rewards)


if __name__ == "__main__":
    unittest.main()
