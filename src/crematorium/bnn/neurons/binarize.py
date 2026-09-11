from __future__ import annotations

from typing import Optional

import torch

from crematorium.module import CrModule, InputTensor, StepOutput, Tensor

from crematorium.bnn.module import (
    _check_word_support,
    _pack_dim,
    _validate_pack_dtype,
)

__all__ = ["XBinarize"]


class XBinarize(CrModule):
    """
    Float-to-binary bridge: thresholds floats at zero (``x > 0`` -> 1)
    and bit-packs along ``dim`` (default last) for packed-binary layers.

    Stateless pass-through complement to the binary engine: use between
    float layers and ``XBLinear``/convolutions instead of feeding floats
    directly (rejected loudly). Non-float inputs are rejected: packing an
    already-packed tensor would silently expand it 8x.
    """

    pack_dtype: torch.dtype = CrModule.Constant(
        torch.uint8, validate=_validate_pack_dtype
    )
    dim: int = CrModule.Constant(default=-1)

    class Specs:
        y = CrModule.OutputSpec(differentiable=False)

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        if not x.is_floating_point():
            raise TypeError(
                f"{type(self).__name__} expects a float tensor, got "
                f"{x.dtype}; it binarizes floats, it does not repack bits"
            )

        bits = torch.iinfo(self.pack_dtype).bits
        return (_pack_dim(x > 0, self.dim, bits, self.pack_dtype),)

    def forward(
        self,
        inputs: InputTensor,
        *state: Tensor,
    ) -> Tensor | StepOutput:
        _check_word_support(self.pack_dtype, inputs, type(self).__name__)
        return super().forward(inputs, *state)

    def forward_sequence(
        self,
        x_seq: InputTensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        _check_word_support(self.pack_dtype, x_seq, type(self).__name__)
        return super().forward_sequence(x_seq, state)
