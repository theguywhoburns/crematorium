from crematorium.module import StepOutput, Tensor
from crematorium.bnn.module import XBModule

__all__ = ["XBLinear"]


class XBLinear(XBModule):
    """
    Packed-binary linear layer.

    ``y = pack(agree(x, W) >= thr)`` with ``agree`` the masked XNOR
    popcount and ``thr = ceil(in_features / 2)`` (majority vote, ties go
    to 1). ``weight`` is a packed ``uint`` buffer ``(out_features,
    in_words)`` holding the actual weights as bits — trained outside the
    module (STE lives outside). The layer itself is a pure boolean
    function: no float params.
    """

    class Specs(XBModule.Specs):
        weight = XBModule.Weight(
            shape=("out_features", "in_words"), min_rank=2, max_rank=2
        )

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        self._cr_check_packed(x)
        agree = self._cr_xnor_agree(x, self.weight)
        return (self._cr_pack_bits(agree >= self._cr_vote_threshold),)
