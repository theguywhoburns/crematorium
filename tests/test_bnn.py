from __future__ import annotations

import pytest
import torch

from crematorium.bnn import XBConv1d, XBConv2d, XBConv3d, XBLinear, XBModule
from crematorium.bnn.module import _swar_byte_counts, _tiled_disagreement

IN, OUT = 130, 17  # non-multiple-of-8 input exercises the tail mask
B, T = 4, 5


def _pack(bits: torch.Tensor) -> torch.Tensor:
    """Independent LSB-first reference packer (not the module under test)."""
    b = bits.to(torch.uint8)
    pad = (-b.shape[-1]) % 8

    if pad:
        b = torch.cat(
            [b, torch.zeros(*b.shape[:-1], pad, dtype=torch.uint8)], dim=-1
        )

    b = b.reshape(*b.shape[:-1], -1, 8)
    return ((b << torch.arange(8, dtype=torch.uint8)).sum(-1)).to(torch.uint8)


def _unpack(packed: torch.Tensor, features: int) -> torch.Tensor:
    b = packed.to(torch.long)
    bits = ((b[..., None] >> torch.arange(8)) & 1).reshape(*b.shape[:-1], -1)
    return bits[..., :features]


def _reference(bx: torch.Tensor, bw: torch.Tensor) -> torch.Tensor:
    """Float-domain XNOR dot thresholded at zero, unpacked 0/1."""
    return (((2 * bx.float() - 1) @ (2 * bw.float() - 1).T) >= 0).to(torch.long)


@pytest.fixture()
def bits():
    torch.manual_seed(0)
    return (
        torch.randint(0, 2, (B, IN)),
        torch.randint(0, 2, (OUT, IN)),
    )


def test_parity_vs_float_reference(bits):
    bx, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT, weight=bw)
    y, _ = m.step_state(_pack(bx), ())
    assert torch.equal(_unpack(y, OUT), _reference(bx, bw))


def test_tail_mask_edges():
    torch.manual_seed(1)

    for n_in in (1, 7, 8, 9):
        bx = torch.randint(0, 2, (B, n_in))
        bw = torch.randint(0, 2, (OUT, n_in))
        m = XBLinear(in_features=n_in, out_features=OUT, weight=bw)
        y, _ = m.step_state(_pack(bx), ())
        assert torch.equal(_unpack(y, OUT), _reference(bx, bw))


def test_packed_weight_accepted_and_persisted(bits):
    bx, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT, weight=_pack(bw))
    assert m.weight.dtype == torch.uint8
    assert tuple(m.weight.shape) == (OUT, 17)
    assert "weight" in m.state_dict()
    y, _ = m.step_state(_pack(bx), ())
    assert torch.equal(_unpack(y, OUT), _reference(bx, bw))


def test_tally_exact_counts(bits):
    bx, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT, weight=bw)
    t = m.tally(_pack(bx))
    assert t.dtype == torch.int32
    assert tuple(t.shape) == (B, OUT)
    ref = torch.stack(
        [torch.stack([(bx[b] == bw[j]).sum() for j in range(OUT)]) for b in range(B)]
    )
    assert torch.equal(t, ref)


def test_tally_threshold_consistent(bits):
    bx, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT, weight=bw)
    xp = _pack(bx)
    thr = (IN + 1) // 2
    y, _ = m.step_state(xp, ())
    assert torch.equal((_unpack(y, OUT) ), ((m.tally(xp) >= thr).to(torch.long)))


def test_tally_validates(bits):
    _, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT, weight=bw)

    with pytest.raises(ValueError, match="packed input"):
        m.tally(torch.zeros(B, 3, dtype=torch.uint8))

    with pytest.raises(ValueError, match="expects 1 inputs"):
        m.tally((_pack(torch.zeros(B, IN, dtype=torch.uint8)),) * 2)


def test_pivotality_predicts_flips():
    # Exhaustive: every sample x channel x bit. Flipping one bit changes
    # the verdict iff the bit is pivotal: agreeing and tally == thr, or
    # disagreeing and tally == thr - 1.
    torch.manual_seed(7)
    n_in, n_out, b = 8, 3, 2
    thr = (n_in + 1) // 2
    bx = torch.randint(0, 2, (b, n_in))
    bw = torch.randint(0, 2, (n_out, n_in))
    m = XBLinear(in_features=n_in, out_features=n_out, weight=bw, validate=False)
    xp = _pack(bx)
    t = m.tally(xp)
    y0, _ = m.step_state(xp, ())
    y0b = _unpack(y0, n_out)

    for bi in range(b):
        for j in range(n_out):
            a = int(t[bi, j])

            for i in range(n_in):
                xf = xp.clone()
                xf[bi, 0] = xf[bi, 0] ^ (1 << i)
                yf, _ = m.step_state(xf, ())
                changed = bool(_unpack(yf, n_out)[bi, j] != y0b[bi, j])
                xbit, wbit = int(bx[bi, i]), int(bw[j, i])
                pivotal = (xbit == wbit and a == thr) or (
                    xbit != wbit and a == thr - 1
                )
                assert changed == pivotal


def test_base_contract():
    assert issubclass(XBLinear, XBModule)
    assert XBModule._cr_output_names == ("y",)
    assert XBModule._cr_state_names == ()


class _Echo(XBModule):
    """Minimal custom layer: Specs + step only, no framework knowledge."""

    class Specs(XBModule.Specs):
        weight = XBModule.Weight(
            shape=("out_features", "in_words"), min_rank=2, max_rank=2
        )

    def _step(self, x):
        self._cr_check_packed(x)
        return (x,)


def test_minimal_subclass_works_end_to_end():
    m = _Echo(in_features=8, out_features=4, weight=torch.zeros(4, 8))
    assert tuple(m.weight.shape) == (4, 1)
    assert "weight" in m.state_dict()
    y, _ = m.step_state(torch.zeros(2, 1, dtype=torch.uint8), ())
    assert torch.equal(y, torch.zeros(2, 1, dtype=torch.uint8))

    with pytest.raises(ValueError, match="Specs"):
        XBModule(in_features=8, out_features=4)


def test_weights_collection_and_signature():
    import inspect

    assert set(XBLinear._cr_weight_specs) == {"weight"}
    assert "weight" in inspect.signature(XBLinear).parameters
    assert "weight" in inspect.signature(XBConv2d).parameters

    m = XBLinear(in_features=8, out_features=4)
    assert m._cr_vote_threshold == 4  # ceil(8 / 2), ties to 1


def test_weights_shape_drift_rejected():
    class _Bad(XBModule):
        class Specs(XBModule.Specs):
            weight = XBModule.Weight(shape=("out_features",), min_rank=2)

        def _step(self, x):
            return (x,)

    with pytest.raises(ValueError, match="min_rank"):
        _Bad(in_features=8, out_features=4)


def test_kernel_int_tuple_equivalence():
    torch.manual_seed(9)
    C, OUT, S = 16, 8, 9
    xb, wb = (
        torch.randint(0, 2, (2, C, S, S)),
        torch.randint(0, 2, (OUT, C, 3, 3)),
    )
    xp = _pack_channels_4d(xb)
    kw = dict(in_features=C, out_features=OUT, weight=wb)
    a, _ = XBConv2d(kernel_size=3, **kw).step_state(xp, ())
    b, _ = XBConv2d(kernel_size=(3, 3), **kw).step_state(xp, ())
    assert torch.equal(a, b)


def _pack_channels_4d(t: torch.Tensor) -> torch.Tensor:
    p = _pack(t.permute(0, 2, 3, 1))
    return p.permute(0, 3, 1, 2)


def test_tie_votes_go_to_one():
    # 8 features, one full word: x = 0x0F, w = 0xFF -> agree = 4 = N/2,
    # margin exactly 0 -> bit 1.
    m = XBLinear(
        in_features=8,
        out_features=1,
        weight=torch.tensor([[0xFF]], dtype=torch.uint8),
    )
    y, _ = m.step_state(torch.tensor([[0x0F]], dtype=torch.uint8), ())
    assert torch.equal(_unpack(y, 1), torch.ones(1, 1, dtype=torch.long))


def test_invalid_construction():
    with pytest.raises(ValueError, match="unsigned integer"):
        XBLinear(in_features=IN, out_features=OUT, pack_dtype=torch.int8)

    with pytest.raises(ValueError, match="positive int"):
        XBLinear(in_features=0, out_features=OUT)

    with pytest.raises(ValueError, match="weight must be"):
        XBLinear(
            in_features=IN,
            out_features=OUT,
            weight=torch.zeros(OUT, 3, dtype=torch.uint8),
        )

    with pytest.raises(ValueError, match="only 0/1"):
        XBLinear(
            in_features=IN,
            out_features=OUT,
            weight=torch.full((OUT, IN), 2),
        )


def test_weight_ranks_declared():
    assert (XBLinear._cr_weight_specs["weight"].min_rank == 2
            and XBLinear._cr_weight_specs["weight"].max_rank == 2)
    assert XBConv1d(in_features=16, out_features=8, kernel_size=3).rank == 1
    assert XBConv2d(in_features=16, out_features=8, kernel_size=3).rank == 2
    assert XBConv3d(in_features=16, out_features=8, kernel_size=3).rank == 3

    with pytest.raises(TypeError, match="not overridable"):
        XBConv2d(in_features=16, out_features=8, kernel_size=3, rank=1)


def test_random_init_breaks_symmetry():
    torch.manual_seed(0)
    a = XBLinear(in_features=IN, out_features=OUT)
    torch.manual_seed(1)
    b = XBLinear(in_features=IN, out_features=OUT)
    assert tuple(a.weight.shape) == (OUT, 17)
    assert a.weight.dtype == torch.uint8
    assert not bool((a.weight == 0).all())
    assert not torch.equal(a.weight, b.weight)


def test_set_weight_validates_like_construction(bits):
    _, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT)
    assert m.set_weight(bw) is m
    ref = XBLinear(in_features=IN, out_features=OUT, weight=bw)
    assert torch.equal(m.weight, ref.weight)

    with pytest.raises(ValueError, match="only 0/1"):
        m.set_weight(torch.full((OUT, IN), 2))

    with pytest.raises(ValueError, match="got None"):
        m.set_weight(None)

    with pytest.raises(ValueError, match="declares no weight 'other'"):
        m.set_weight(bw, name="other")


def test_float_input_rejected_loudly():
    m = XBLinear(in_features=IN, out_features=OUT)

    with pytest.raises(TypeError, match="XBinarize"):
        m.step_state(torch.randn(B, IN), ())


def test_xbinarize_threshold_and_pack():
    from crematorium.bnn import XBinarize

    m = XBinarize()
    x = torch.tensor([[-1.0, 0.0, 0.5, 2.0, -0.0, 1.0, -3.0, 4.0, 0.25]])
    y, _ = m.step_state(x, ())
    assert tuple(y.shape) == (1, 2)
    assert y.dtype == torch.uint8
    # (x > 0): zeros (incl. -0.0) map to 0-bits, positives to 1-bits.
    assert torch.equal(y, torch.tensor([[0b10101100, 0b00000001]], dtype=torch.uint8))


def test_xbinarize_channel_dim():
    from crematorium.bnn import XBinarize

    torch.manual_seed(3)
    m = XBinarize(dim=1)
    xb = torch.randint(0, 2, (B, 16, 5, 5)).float() * 2 - 1
    y, _ = m.step_state(xb, ())
    assert tuple(y.shape) == (B, 2, 5, 5)
    assert y.dtype == torch.uint8


def test_xbinarize_rejects_non_float():
    from crematorium.bnn import XBinarize

    m = XBinarize()

    with pytest.raises(TypeError, match="expects a float"):
        m.step_state(torch.zeros(2, 8, dtype=torch.uint8), ())


def test_wrong_word_count_input_raises(bits):
    _, bw = bits
    m = XBLinear(in_features=IN, out_features=OUT, weight=bw)

    with pytest.raises(ValueError, match="packed input"):
        m.step_state(torch.zeros(B, 3, dtype=torch.uint8), ())


def test_forward_sequence_matches_steps(bits):
    bx, bw = bits
    x_seq = _pack(bx).unsqueeze(0).expand(T, B, -1).clone()
    m = XBLinear(in_features=IN, out_features=OUT, weight=bw, validate=False)

    seq = m.forward_sequence(x_seq)
    seq = seq[0] if isinstance(seq, tuple) else seq

    outs = [m.step_state(x_seq[t], ())[0] for t in range(T)]
    assert torch.equal(seq, torch.stack(outs))


def test_compiled_matches_eager(bits):
    torch._dynamo.reset()
    bx, bw = bits
    x_seq = _pack(bx).unsqueeze(0).expand(T, B, -1).clone()

    ref = XBLinear(in_features=IN, out_features=OUT, weight=bw, validate=False)
    expected = ref.forward_sequence(x_seq)
    expected = expected[0] if isinstance(expected, tuple) else expected

    m = XBLinear(in_features=IN, out_features=OUT, weight=bw, validate=False)
    m.fast_sequence_()
    got = m.forward_sequence(x_seq)
    got = got[0] if isinstance(got, tuple) else got
    assert torch.equal(got, expected)


def test_sequential_two_xblinear(bits):
    from crematorium.nn import Sequential

    torch.manual_seed(2)
    bx, bw = bits
    bw2 = torch.randint(0, 2, (5, OUT))
    x_seq = _pack(bx).unsqueeze(0).expand(T, B, -1).clone()

    def make():
        return Sequential(
            XBLinear(in_features=IN, out_features=OUT, weight=bw, validate=False),
            XBLinear(in_features=OUT, out_features=5, weight=bw2, validate=False),
            init_hidden=False,
        )

    net = make()
    seq = net.forward_sequence(x_seq)

    l1, l2 = net[0], net[1]
    outs = []

    for t in range(T):
        y1, _ = l1.step_state(x_seq[t], ())
        y2, _ = l2.step_state(y1, ())
        outs.append(y2)

    assert torch.equal(seq[0] if isinstance(seq, tuple) else seq, torch.stack(outs))

    torch._dynamo.reset()
    fast = make()
    fast.fast_sequence_()
    got = fast.forward_sequence(x_seq)
    got = got[0] if isinstance(got, tuple) else got
    assert torch.equal(got, torch.stack(outs))


# ----------------------------------------------------------------------
# Double-tiled disagreement kernel
# ----------------------------------------------------------------------


def _single_shot(xm: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Untiled reference: one broadcast XOR + popcount (small inputs only)."""
    xor = xm[:, None, :] ^ w[None, :, :]
    counts = _swar_byte_counts(xor, 8)
    return counts.reshape(*counts.shape[:-2], -1).sum(-1, dtype=torch.int32)


def test_tiled_disagreement_matches_single_shot():
    torch.manual_seed(0)
    # Non-divisible tiles on both axes, plus tail padding (Wp=17).
    xm = torch.randint(0, 256, (100, 17), dtype=torch.uint8)
    w = torch.randint(0, 256, (100, 17), dtype=torch.uint8)
    got = _tiled_disagreement(xm, w, 8, 48, 64, 1 << 26)
    assert torch.equal(got, _single_shot(xm, w))


def test_tiled_disagreement_small_is_single_shot():
    torch.manual_seed(1)
    xm = torch.randint(0, 256, (4, 17), dtype=torch.uint8)
    w = torch.randint(0, 256, (8, 17), dtype=torch.uint8)
    got = _tiled_disagreement(xm, w, 8, 256, 4096, 1 << 26)
    assert torch.equal(got, _single_shot(xm, w))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="peak assertion needs CUDA"
)
def test_tiled_disagreement_bounds_peak():
    torch.manual_seed(2)
    M, Wp, m = 20000, 72, 512
    xm = torch.randint(0, 256, (M, Wp), dtype=torch.uint8, device="cuda")
    w = torch.randint(0, 256, (m, Wp), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    got = _tiled_disagreement(xm, w, 8, 256, 4096, 1 << 26)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    # Untiled, the XOR alone is 20000*256*72 = 368MB plus ~2GB of
    # follow-on transients; the block budget is 64MB.
    assert peak < 512, f"peak {peak:.0f}MB exceeds tiled budget"
    assert torch.equal(got[:100].cpu(), _single_shot(xm[:100].cpu(), w.cpu()))
