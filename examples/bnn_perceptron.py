"""Minimal true-binary training loop (CPU, seconds).

No float shadows, no autograd: the engine is integer-pure, so training
is discrete. The harness owns unpacked binary weights; on each mistake
it flips the disagreeing bits toward the input (binary perceptron rule,
with disagreement located via ``tally``) and injects them through
``set_weight``, which validates like construction.
"""

import torch

from crematorium.bnn import XBinarize, XBLinear


def unpack(packed: torch.Tensor, features: int) -> torch.Tensor:
    b = packed.to(torch.long)
    bits = ((b[..., None] >> torch.arange(8)) & 1).reshape(*b.shape[:-1], -1)
    return bits[..., :features]


def main() -> None:
    torch.manual_seed(0)
    IN, OUT, N = 64, 8, 512

    teacher = torch.randint(0, 2, (OUT, IN))
    xb = torch.randint(0, 2, (N, IN))
    # Majority-vote labels from the teacher rows.
    yb = (((xb[:, None, :] == teacher[None]).sum(-1) * 2 - IN) >= 0).long()

    engine = XBLinear(in_features=IN, out_features=OUT)
    encode = XBinarize()
    packed_xb, _ = encode.step_state(xb.float() * 2 - 1, ())

    def error_rate() -> float:
        out, _ = engine.step_state(packed_xb, ())
        return (unpack(out, OUT) != yb).float().mean().item()

    init_err = error_rate()
    print(f"init error: {init_err:.3f}")

    # Harness-owned binary weights, injected validated. Pocket-best is
    # kept because greedy discrete updates oscillate (standard practice
    # for binary training).
    bits = torch.randint(0, 2, (OUT, IN))
    engine.set_weight(bits)
    thr = (IN + 1) // 2
    best_bits, best_err = bits.clone(), init_err

    for step in range(400):
        idx = torch.randint(0, N, (32,))
        pred = (engine.tally(packed_xb[idx]) >= thr).long()
        wrong = pred != yb[idx]

        if wrong.any():
            disagree = (xb[idx].unsqueeze(1) != bits.unsqueeze(0)) & wrong.unsqueeze(-1)
            bits ^= disagree.any(dim=0)
            engine.set_weight(bits)

        if step % 25 == 0:
            e = error_rate()

            if e < best_err:
                best_err, best_bits = e, bits.clone()

    engine.set_weight(best_bits)
    err = error_rate()
    print(f"final error: {err:.3f}")
    assert err < init_err, (init_err, err)


if __name__ == "__main__":
    main()
