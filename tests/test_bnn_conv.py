from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from crematorium.bnn import XBConv1d, XBConv2d, XBConv3d, XBCnn

B = 2


def _pack_last(bits: torch.Tensor) -> torch.Tensor:
    b = bits.to(torch.uint8)
    pad = (-b.shape[-1]) % 8

    if pad:
        b = torch.cat(
            [b, torch.zeros(*b.shape[:-1], pad, dtype=torch.uint8)], dim=-1
        )

    b = b.reshape(*b.shape[:-1], -1, 8)
    return ((b << torch.arange(8, dtype=torch.uint8)).sum(-1)).to(torch.uint8)


def _unpack_last(packed: torch.Tensor, features: int) -> torch.Tensor:
    b = packed.to(torch.long)
    bits = ((b[..., None] >> torch.arange(8)) & 1).reshape(*b.shape[:-1], -1)
    return bits[..., :features]


def _pack_channels(t: torch.Tensor) -> torch.Tensor:
    d = t.dim()
    p = _pack_last(t.permute(0, *range(2, d), 1))
    return p.permute(0, d - 1, *range(1, d - 1))


def _unpack_channels(packed: torch.Tensor, channels: int) -> torch.Tensor:
    d = packed.dim()
    u = _unpack_last(packed.permute(0, *range(2, d), 1), channels)
    return u.permute(0, d - 1, *range(1, d - 1))


def _bits(*shape):
    return torch.randint(0, 2, shape)


def test_conv1d_parity():
    torch.manual_seed(0)
    C, OUT, L, K = 16, 8, 21, 3
    xb, wb = _bits(B, C, L), _bits(OUT, C, K)
    m = XBConv1d(in_features=C, out_features=OUT, kernel_size=K, weight=wb)
    y, _ = m.step_state(_pack_channels(xb), ())
    ref = F.conv1d(2 * xb.float() - 1, 2 * wb.float() - 1) >= 0
    assert torch.equal(_unpack_channels(y, OUT), ref.to(torch.long))


def test_conv2d_parity():
    torch.manual_seed(1)
    C, OUT, S, K = 16, 8, 9, 3
    xb, wb = _bits(B, C, S, S), _bits(OUT, C, K, K)
    m = XBConv2d(in_features=C, out_features=OUT, kernel_size=K, weight=wb)
    y, _ = m.step_state(_pack_channels(xb), ())
    ref = F.conv2d(2 * xb.float() - 1, 2 * wb.float() - 1) >= 0
    assert torch.equal(_unpack_channels(y, OUT), ref.to(torch.long))


def test_conv3d_parity():
    torch.manual_seed(2)
    C, OUT, S, K = 8, 4, 5, 3
    xb, wb = _bits(B, C, S, S, S), _bits(OUT, C, K, K, K)
    m = XBConv3d(in_features=C, out_features=OUT, kernel_size=K, weight=wb)
    y, _ = m.step_state(_pack_channels(xb), ())
    ref = F.conv3d(2 * xb.float() - 1, 2 * wb.float() - 1) >= 0
    assert torch.equal(_unpack_channels(y, OUT), ref.to(torch.long))


def test_conv2d_stride_padding_bitfaithful():
    # Padding pads packed bits with 0 (a vote, not neutral): replicate the
    # same bit-level semantics in the float reference to pin the behavior.
    torch.manual_seed(3)
    C, OUT, S, K, ST, P = 16, 8, 9, 3, 2, 1
    xb, wb = _bits(B, C, S, S), _bits(OUT, C, K, K)
    m = XBConv2d(
        in_features=C, out_features=OUT, kernel_size=K, stride=ST, padding=P,
        weight=wb,
    )
    y, _ = m.step_state(_pack_channels(xb), ())
    xp = F.pad(xb, (P, P, P, P))
    ref = F.conv2d(2 * xp.float() - 1, 2 * wb.float() - 1, stride=ST) >= 0
    assert tuple(y.shape[2:]) == (5, 5)
    assert torch.equal(_unpack_channels(y, OUT), ref.to(torch.long))


def test_ranks_and_generic_guard():
    assert XBConv1d._cr_weight_specs["weight"].min_rank == 3
    assert XBConv2d._cr_weight_specs["weight"].min_rank == 3
    assert XBConv3d._cr_weight_specs["weight"].min_rank == 3
    assert XBConv3d._cr_weight_specs["weight"].max_rank == 5

    with pytest.raises(ValueError, match="rank must be"):
        XBCnn(in_features=16, out_features=8, kernel_size=3)

    with pytest.raises(ValueError, match="rank must be"):
        XBCnn(rank=5, in_features=16, out_features=8, kernel_size=3)


def test_direct_rank_construction_matches_alias():
    torch.manual_seed(11)
    C, OUT, S, K = 16, 8, 9, 3
    xb, wb = _bits(B, C, S, S), _bits(OUT, C, K, K)
    xp = _pack_channels(xb)
    kw = dict(in_features=C, out_features=OUT, kernel_size=K, weight=wb,
              validate=False)
    ref, _ = XBConv2d(**kw).step_state(xp, ())
    got, _ = XBCnn(rank=2, **kw).step_state(xp, ())
    assert torch.equal(got, ref)


def test_channel_and_kernel_validation():
    with pytest.raises(ValueError, match="multiple of 8"):
        XBConv2d(in_features=7, out_features=8, kernel_size=3)

    with pytest.raises(ValueError, match="length"):
        XBConv2d(in_features=16, out_features=8, kernel_size=(3, 3, 3))

    with pytest.raises(ValueError, match=">= 1"):
        XBConv2d(in_features=16, out_features=8, kernel_size=0)


def test_spatial_misuse_rejected():
    from crematorium.bnn import Spatial

    with pytest.raises(ValueError, match="non-negative int"):
        Spatial(minimum=-1)


def test_weight_forms():
    torch.manual_seed(4)
    C, OUT, S, K = 16, 8, 9, 3
    xb, wb = _bits(B, C, S, S), _bits(OUT, C, K, K)
    xp = _pack_channels(xb)
    ref, _ = XBConv2d(
        in_features=C, out_features=OUT, kernel_size=K, weight=wb
    ).step_state(xp, ())

    m = XBConv2d(in_features=C, out_features=OUT, kernel_size=K,
                 weight=_pack_channels(wb))
    assert m.weight.dtype == torch.uint8
    assert tuple(m.weight.shape) == (OUT, C // 8, K, K)
    assert "weight" in m.state_dict()
    got, _ = m.step_state(xp, ())
    assert torch.equal(got, ref)

    with pytest.raises(ValueError, match="rank"):
        XBConv2d(in_features=C, out_features=OUT, kernel_size=K,
                 weight=torch.zeros(OUT, C, K, K, 1, dtype=torch.uint8))


def test_tally_shape_and_consistency():
    torch.manual_seed(5)
    C, OUT, S, K = 16, 8, 9, 3
    xb, wb = _bits(B, C, S, S), _bits(OUT, C, K, K)
    m = XBConv2d(in_features=C, out_features=OUT, kernel_size=K, weight=wb)
    xp = _pack_channels(xb)
    t = m.tally(xp)
    assert tuple(t.shape) == (B, OUT, S - K + 1, S - K + 1)
    thr = (C * K * K + 1) // 2
    y, _ = m.step_state(xp, ())
    assert torch.equal((t >= thr).to(torch.long), _unpack_channels(y, OUT))


def test_sequence_and_compiled():
    torch.manual_seed(6)
    C, OUT, S, K, T = 16, 8, 9, 3, 4
    xb = _bits(T, B, C, S, S)
    wb = _bits(OUT, C, K, K)
    xp = torch.stack([_pack_channels(xb[t]) for t in range(T)])

    def make():
        return XBConv2d(in_features=C, out_features=OUT, kernel_size=K,
                        weight=wb, validate=False)

    m = make()
    seq = m.forward_sequence(xp)
    seq = seq[0] if isinstance(seq, tuple) else seq
    outs = [make().step_state(xp[t], ())[0] for t in range(T)]
    assert torch.equal(seq, torch.stack(outs))

    torch._dynamo.reset()
    fast = make()
    fast.fast_sequence_()
    got = fast.forward_sequence(xp)
    got = got[0] if isinstance(got, tuple) else got
    assert torch.equal(got, torch.stack(outs))


def test_row_tiling_matches_untiled():
    # Force the M-tiling path on a small map via an instance-level tile
    # override (class default untouched): M=162 rows over 64-row blocks.
    torch.manual_seed(7)
    C, OUT, S, K = 16, 8, 9, 3
    xb, wb = _bits(B, C, S, S), _bits(OUT, C, K, K)
    xp = _pack_channels(xb)

    ref, _ = XBConv2d(in_features=C, out_features=OUT, kernel_size=K,
                      weight=wb, validate=False).step_state(xp, ())
    m = XBConv2d(in_features=C, out_features=OUT, kernel_size=K,
                 weight=wb, validate=False)
    m._TILE_ROWS = 64
    m._TILE_OUTPUTS = 5
    got, _ = m.step_state(xp, ())
    assert torch.equal(got, ref)
    del m._TILE_ROWS, m._TILE_OUTPUTS
    assert XBConv2d._TILE_ROWS == 4096
