# SPDX-License-Identifier: Apache-2.0
"""New ASR example, not a reward recipe extracted from ms-swift."""
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not unicodedata.category(c).startswith("P"))


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def recognition_reward(hypothesis, reference, unit="char"):
    if unit not in {"char", "word"}:
        raise ValueError("unit must be char or word")
    def tokens(text):
        normalized = normalize(text)
        return normalized.split() if unit == "word" else list("".join(normalized.split()))
    hyp, ref = tokens(hypothesis), tokens(reference)
    if not ref:
        # Explicit pure-noise/silence behavior; no division by zero.
        return 1.0 if not hyp else -1.0
    return 1.0 - edit_distance(ref, hyp) / len(ref)


def asr_rewards(batch, texts, group_size=4, unit="char"):
    """batch: one dict with `reference` per utterance, contiguous G completions."""
    references = [row["reference"] for row in batch for _ in range(group_size)]
    if len(references) != len(texts):
        raise ValueError("Expected len(batch) * group_size completions")
    return [[recognition_reward(h, r, unit)] for h, r in zip(texts, references)], [1.0]
