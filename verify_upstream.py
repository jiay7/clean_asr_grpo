"""Optional online parity check against a pinned upstream function.

Executes only the selected compute_advantages function from the pinned source.
Run from parent directory: python -m clean_asr_grpo.verify_upstream
"""
import ast
import typing
import urllib.request
import torch
from .core import Config, advantages

COMMIT = "f7385b200c4f629dec1a70af32c9b4174067b9c1"


def main():
    url = f"https://raw.githubusercontent.com/modelscope/ms-swift/{COMMIT}/swift/rl_core/advantage.py"
    with urllib.request.urlopen(url, timeout=30) as response:
        source = response.read().decode()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "compute_advantages")
    namespace = {"torch": torch, "Optional": typing.Optional, "Tuple": typing.Tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), url, "exec"), namespace)
    torch.manual_seed(42)
    for group_size in [2, 4, 8]:
        for scale in ["group", "batch", "none"]:
            rewards = torch.randn(3 * group_size, 3)
            rewards[0, 1] = float("nan")
            weights = torch.tensor([1., 0.3, -0.1])
            actual = advantages(rewards, weights, Config(group_size=group_size, scale_rewards=scale))
            expected = namespace["compute_advantages"](rewards, weights, group_size, scale_rewards=scale)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("PASS: advantages match pinned upstream exactly in 9 configurations")


if __name__ == "__main__":
    main()
