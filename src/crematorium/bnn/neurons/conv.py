import torch.nn.functional as F

from crematorium.module import InputTensor, StepOutput, Tensor
from crematorium.util import (
    multiple_of,
    spatial_rank,
)

from crematorium.bnn.module import XBModule, _tiled_disagreement

__all__ = [
    "XBCnn",
    "XBConv1d",
    "XBConv2d",
    "XBConv3d",
]


# Shared weight layout: channel-words second, kernel taps splat to the
# subclass's spatial rank. Declared per dim (with its rank) below.
_CONV_WEIGHT_SHAPE = ("out_features", "in_words", "kernel_size")


class XBCnn(XBModule):
    """
    Base for packed-binary XNOR convolutions.

    Same contract as the linear layer, over sliding windows: each output
    position tallies agreement between its unpacked patch (``in_features``
    channels x kernel taps) and each filter, thresholds at majority, and
    repacks. ``in_features``/``out_features`` are channel counts;
    ``in_features`` must be a multiple of 8 (partial channel words would
    need a per-tap tail mask; standard channel counts already qualify).
    ``rank`` is the spatial rank (1, 2, or 3): pass it to construct any
    dimensionality directly — ``XBCnn(rank=1, ...)`` — or use the
    ``XBConvNd`` aliases, which preconfigure it.

    Padding pads packed bits with 0, i.e. border positions vote toward
    matching weight-0 taps — there is no neutral bit. This matches
    XNOR-Net's own sign(0) edge convention. The default ``padding=0`` is
    unaffected. Dilation and groups are unsupported.
    """

    in_features: int = XBModule.Constant(dtype=int, validate=multiple_of(8))

    kernel_size: tuple[int, ...] = XBModule.Spatial(minimum=1)
    stride: tuple[int, ...] = XBModule.Spatial(default=1, minimum=1)
    padding: tuple[int, ...] = XBModule.Spatial(default=0)
    rank: int = XBModule.Constant(validate=spatial_rank)

    class Specs(XBModule.Specs):
        weight = XBModule.Weight(
            shape=_CONV_WEIGHT_SHAPE, min_rank=3, max_rank=5
        )

    def _extract_patches(self, x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        """
        ``(B, Cw, *S)`` -> (``(B*L, Cw*prod(K))`` patches, output spatial).

        Fully determined by the normalized kernel/stride/padding: pad
        once, unfold each spatial dim in turn (views, dtype-agnostic —
        ``F.unfold`` has no uint8 kernel), then move channels+taps
        innermost. Layout is (channel, taps) by construction, matching
        the C-order flatten of the weight.
        """
        kernel, stride, padding = self.kernel_size, self.stride, self.padding
        spatial = len(kernel)
        xp = x

        if any(padding):
            xp = F.pad(xp, tuple(v for p in reversed(padding) for v in (p, p)))

        for i in range(spatial):
            xp = xp.unfold(2 + i, kernel[i], stride[i])

        out_spatial = tuple(xp.shape[2 : 2 + spatial])
        u = xp.permute(
            0, *range(2, 2 + spatial), 1, *range(2 + spatial, 2 + 2 * spatial)
        )
        return u.flatten(0, spatial).flatten(1), out_spatial

    def tally(self, inputs: InputTensor) -> Tensor:
        """
        Raw agreement tallies, ``(B, O, *S_out)`` int32: per-(sample,
        channel, position) count of bit positions where patch and filter
        point the same way.
        """
        x = self._cr_canonicalize_inputs(inputs)[0]

        if x.dtype != self.pack_dtype:
            raise TypeError(
                f"{type(self).__name__} expects packed {self.pack_dtype} "
                f"input, got {x.dtype}; binarize floats first "
                f"(see XBinarize)"
            )

        if x.dim() != self.weight.dim() or x.shape[1] != self._cr_in_words():
            raise ValueError(
                f"{type(self).__name__} expects packed (B, Cw, *S) with "
                f"{self._cr_in_words()} channel words "
                f"({self.in_features} channels), got shape {tuple(x.shape)}"
            )

        patches, out_spatial = self._extract_patches(x)
        w = self.weight.reshape(self.out_features, -1)
        dis = _tiled_disagreement(
            patches,
            w,
            self._cr_word_bits(),
            self._TILE_OUTPUTS,
            self._TILE_ROWS,
            self._TILE_BYTES,
        )
        agree = self._cr_n_votes - dis
        return (
            agree.reshape(x.shape[0], -1, self.out_features)
            .transpose(1, 2)
            .reshape(x.shape[0], self.out_features, *out_spatial)
        )

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        return (self._cr_pack_channels(self.tally(x) >= self._cr_vote_threshold),)


class XBConv1d(XBCnn):
    """Packed-binary 1D convolution over ``(B, C, L)``."""

    rank: int = XBCnn.Constant(
        default=1, validate=spatial_rank, overridable=False
    )


class XBConv2d(XBCnn):
    """Packed-binary 2D convolution over ``(B, C, H, W)``."""

    rank: int = XBCnn.Constant(
        default=2, validate=spatial_rank, overridable=False
    )


class XBConv3d(XBCnn):
    """Packed-binary 3D convolution over ``(B, C, D, H, W)``."""

    rank: int = XBCnn.Constant(
        default=3, validate=spatial_rank, overridable=False
    )
