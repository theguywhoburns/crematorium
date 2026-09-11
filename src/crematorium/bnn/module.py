from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import torch

from crematorium.module import (
    ConstantSpec,
    CrModule,
    InputTensor,
    StepOutput,
    Tensor,
)
from crematorium.util import normalize_spatial, spatial_spec

__all__ = [
    "Weight",
    "WeightSpec",
    "XBModule",
]


_UINT_WORD_DTYPES = (torch.uint8, torch.uint16, torch.uint32, torch.uint64)

# (pack_dtype, device.type) -> shift support. Probed lazily once; see
# XBModule._cr_check_word_support.
_WORD_SUPPORT_CACHE: dict[tuple[torch.dtype, str], bool] = {}


def _validate_pack_dtype(value: Any) -> None:
    if not isinstance(value, torch.dtype) or value not in _UINT_WORD_DTYPES:
        raise ValueError(
            f"pack_dtype must be an unsigned integer torch dtype "
            f"{[str(d) for d in _UINT_WORD_DTYPES]}, got {value!r}"
        )


def _validate_dim(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"must be a positive int, got {value!r}")


def _swar_byte_counts(x: Tensor, bits: int) -> Tensor:
    """
    Per-byte population counts, fixed-constant SWAR.

    Words are split into bytes with one arange shift (a trailing size-1
    dim for 8-bit words); each byte is counted with constant masks. No
    Python loops, no tables, no reduction here: the caller sums words and
    bytes in a single kernel, so no ``(B, O, Wp)`` int64 intermediate ever
    materializes. Traces under ``torch.compile`` and runs under
    ``FakeTensorMode``. For 8-bit words the split is degenerate (single
    size-1 dim), so the branch below counts the words directly and skips
    the shift/mask passes entirely.
    """
    if bits == 8:
        # Degenerate split: one size-1 dim (a free view, no copy). The
        # shift/mask passes are skipped; SWAR runs on the words directly.
        b = x[..., None]
    else:
        shifts = torch.arange(0, bits, 8, device=x.device, dtype=x.dtype)
        b = (x[..., None] >> shifts) & 0xFF

    b = b - ((b >> 1) & 0x55)
    b = (b & 0x33) + ((b >> 2) & 0x33)
    return (b + (b >> 4)) & 0x0F


def _tiled_disagreement(
    xm: Tensor, w: Tensor, bits: int, tile_out: int, tile_m: int, tile_bytes: int
) -> Tensor:
    """
    ``(M, Wp)``, ``(m, Wp)`` -> ``(M, m)`` int32 disagreement counts.

    Shared kernel behind the linear and convolutional agreement paths:
    single shot under ``torch.compile`` (loops would unroll into
    per-block launches plus a concat there). In eager mode both axes
    are tiled — for convolutions the rows are batch x positions, so an
    untiled M grows the transient with feature-map size. The row tile
    shrinks so a block never exceeds ``tile_bytes``. Blocks write into
    one preallocated output (no cat-chains); a single block takes the
    allocation-free fast path. Inputs arrive canonical (padding zeroed);
    no masking in here.
    """
    if torch.compiler.is_compiling():
        counts = _swar_byte_counts(xm[:, None, :] ^ w[None, :, :], bits)
        return counts.reshape(*counts.shape[:-2], -1).sum(-1, dtype=torch.int32)

    M, m = xm.shape[0], w.shape[0]
    nbytes = bits // 8
    rows = max(1, min(tile_m, tile_bytes // max(1, tile_out * xm.shape[-1] * nbytes)))

    if M <= rows and m <= tile_out:
        xor = xm[:, None, :] ^ w[None, :, :]
        counts = _swar_byte_counts(xor, bits)
        return counts.reshape(*counts.shape[:-2], -1).sum(-1, dtype=torch.int32)

    out = torch.empty(M, m, dtype=torch.int32, device=xm.device)

    for m_lo in range(0, M, rows):
        m_hi = min(m_lo + rows, M)
        xblk = xm[m_lo:m_hi]

        for o_lo in range(0, m, tile_out):
            o_hi = min(o_lo + tile_out, m)
            xor = xblk[:, None, :] ^ w[o_lo:o_hi][None, :, :]
            counts = _swar_byte_counts(xor, bits)
            out[m_lo:m_hi, o_lo:o_hi] = counts.reshape(
                *counts.shape[:-2], -1
            ).sum(-1, dtype=torch.int32)

    return out


def _first_tensor(t: Any) -> Optional[Tensor]:
    if isinstance(t, Tensor):
        return t

    if isinstance(t, (tuple, list)):
        for v in t:
            found = _first_tensor(v)
            if found is not None:
                return found

        return None

    if isinstance(t, dict):
        for v in t.values():
            found = _first_tensor(v)
            if found is not None:
                return found

    return None


@dataclass(frozen=True)
class SpatialSpec(ConstantSpec):
    """
    A ``Constant`` holding a spatial spec, normalized to a rank-tuple at
    construction. Declared via ``Spatial``.
    """

    minimum: int = 0


def Spatial(default: Any = None, *, minimum: int = 0) -> Any:
    """
    Declarative spatial-spec field (hides the ``Constant`` boilerplate)::

        kernel_size: tuple[int, ...] = XBModule.Spatial(minimum=1)

    A bare ``default`` of None means required (rejected by validation),
    matching ``Constant`` semantics. Returns Any: annotate the field
    with its normalized (use-time) type.
    """
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError(
            f"Spatial minimum must be a non-negative int, got {minimum!r}"
        )

    return SpatialSpec(
        default=default, validate=spatial_spec, dtype=None, minimum=minimum
    )


@dataclass(frozen=True)
class WeightSpec:
    """
    Declares one packed-binary weight buffer on the module, inside the
    regular ``Specs`` class (the engine ignores it: it is neither output
    nor recurrent state).

    ``shape`` entries are ints (literal dims) or names. A name resolving
    to an int Constant is one dim; a name resolving to an int tuple
    (e.g. a normalized spatial spec) splats into several. ``"in_words"``
    resolves to the derived channel-word count. Anything else raises a
    clear error at construction.

    ``rank`` is replaced by ``min_rank``/``max_rank`` bounds on the
    resolved buffer ndim (None = unbounded): an exact pair pins a
    topology, a range shares one declaration across dimensionalities.
    A resolved shape outside the bounds raises at construction.
    """

    shape: tuple = ()
    min_rank: Optional[int] = None
    max_rank: Optional[int] = None


def _check_rank_bound(value: Any, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"Weight {name} must be an int or None, got {value!r}")


def Weight(
    *,
    shape: tuple = (),
    min_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
) -> Any:
    """
    Declarative packed-binary weight field, inside ``Specs``::

        class Specs(XBModule.Specs):
            weight = XBModule.Weight(
                shape=("out_features", "in_words", "kernel_size"),
                min_rank=3,
                max_rank=5,
            )

    Returns Any (the buffer itself is registered at construction).
    """
    _check_rank_bound(min_rank, "min_rank")
    _check_rank_bound(max_rank, "max_rank")

    if min_rank is not None and max_rank is not None and min_rank > max_rank:
        raise ValueError(
            f"Weight min_rank ({min_rank}) exceeds max_rank ({max_rank})"
        )

    return WeightSpec(shape=tuple(shape), min_rank=min_rank, max_rank=max_rank)


def _collect_weights(cls: type) -> dict[str, WeightSpec]:
    """Collect WeightSpecs from nested ``Specs`` classes across the MRO."""
    specs: dict[str, WeightSpec] = {}

    for klass in reversed(cls.__mro__):
        specs_cls = klass.__dict__.get("Specs", None)

        if specs_cls is None:
            continue

        for name, value in vars(specs_cls).items():
            if isinstance(value, WeightSpec):
                specs[name] = value

    return specs


def _pack_bits_last(t: Tensor, bits: int, dtype: torch.dtype) -> Tensor:
    """``(..., F)`` {0,1}/bool -> ``(..., ceil(F / bits))`` packed LSB-first."""
    f = t.shape[-1]
    pad = (-f) % bits
    b = t.to(dtype)

    if pad:
        b = torch.cat(
            [b, torch.zeros(*b.shape[:-1], pad, dtype=b.dtype, device=b.device)],
            dim=-1,
        )

    b = b.reshape(*b.shape[:-1], -1, bits)
    shifts = torch.arange(bits, dtype=dtype, device=b.device)
    # sum() upcasts integers to int64; cast back to the word dtype.
    return ((b << shifts).sum(-1)).to(dtype)


def _pack_dim(t: Tensor, dim: int, bits: int, dtype: torch.dtype) -> Tensor:
    """Pack binary values along ``dim`` (negative ok), LSB-first."""
    d = t.dim()
    dim %= d
    order = [i for i in range(d) if i != dim] + [dim]
    p = _pack_bits_last(t.permute(*order), bits, dtype)
    inv = [0] * d

    for j, orig in enumerate(order):
        inv[orig] = j

    return p.permute(*inv)


def _check_word_support(
    dtype: torch.dtype, inputs: InputTensor, owner: str
) -> None:
    """Probe bitwise-shift support for a pack dtype, cached per device."""
    if dtype == torch.uint8:
        return

    sample = _first_tensor(inputs)

    if sample is None:
        return

    try:
        from torch._subclasses.fake_tensor import (  # pyright: ignore[reportMissingImports]
            FakeTensor,
        )
    except ImportError:
        FakeTensor = None  # type: ignore[assignment]

    if FakeTensor is not None and isinstance(sample, FakeTensor):
        return

    key = (dtype, sample.device.type)
    ok = _WORD_SUPPORT_CACHE.get(key)

    if ok is None:
        try:
            t = torch.ones(1, dtype=dtype, device=sample.device)
            _ = (t ^ t) >> 1
            ok = True
        except RuntimeError:
            ok = False

        _WORD_SUPPORT_CACHE[key] = ok

    if not ok:
        raise RuntimeError(
            f"{owner}: pack_dtype={dtype} has no "
            f"bitwise-shift support on {sample.device.type}; use torch.uint8"
        )


class XBModule(CrModule):
    """
    Base for packed-binary XNOR layers.

    Inputs and outputs are binary values bit-packed LSB-first (feature ``i``
    lives in bit ``i``) into unsigned integer words. Nothing is unpacked on
    the hot path: the dot product runs in the packed domain as
    ``2 * popcount(XNOR) - N``. Surrogate gradients / STE live outside the
    module; ``_step`` is pure binary math.

    Generic dims live here; subclasses add only their topology (weight
    layout, extra geometry constants, ``_step``):

        in_features   unpacked input width (required positive int)
        out_features  unpacked output width (required positive int)
        pack_dtype    unsigned word dtype (default ``torch.uint8``); word
                      width in bits is derived via ``torch.iinfo``

    Declaring weights: alongside outputs, a layer declares its packed
    weight buffers in the regular ``Specs`` class via ``Weight`` fields::

        class Specs(XBModule.Specs):
            weight = XBModule.Weight(
                shape=("out_features", "in_words", "kernel_size"),
                min_rank=3,
                max_rank=5,
            )

    Spatial geometry (kernel/stride/padding) is declared with ``Spatial``,
    which hides the ``Constant`` boilerplate and normalizes values to
    rank-tuples at construction::

        kernel_size: tuple[int, ...] = XBModule.Spatial(minimum=1)

    Shape entries are ints (literal dims) or names: a name resolving to an
    int Constant is one dim, a name resolving to an int tuple (e.g. a
    normalized spatial spec) splats into several, and ``"in_words"``
    resolves to the derived channel-word count. ``min_rank``/``max_rank``
    bound the resolved buffer ndim (an exact pair pins a topology, a
    range shares one declaration across dimensionalities). The framework
    collects the fields across the MRO (child wins), validates/coerces
    each value (packed, unpacked 0/1, or None for zeros), registers
    persistent buffers, and derives the vote threshold from the
    registered shapes.

    Math (per output channel ``j``, ``N`` = votes)::

        agree_j = popcount(XNOR(x_packed, W_packed[j]))  over valid bits,
                  computed as N - popcount(XOR) so no flip pass is needed
        y_j     = agree_j >= thr, thr = ceil(N / 2)   (majority vote, ties to 1)
    """

    # Namespaced declarative helpers. staticmethod (not bare alias): bare
    # assignments would make checkers bind them as methods.
    Weight = staticmethod(Weight)
    Spatial = staticmethod(Spatial)

    # Collected WeightSpecs per class (child wins); filled by
    # __init_subclass__ before CrModule's own hook runs (the signature
    # generator reads it).
    _cr_weight_specs: ClassVar[dict[str, WeightSpec]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls._cr_weight_specs = _collect_weights(cls)
        super().__init_subclass__(**kwargs)

    in_features: int = CrModule.Constant(dtype=int, validate=_validate_dim)
    out_features: int = CrModule.Constant(dtype=int, validate=_validate_dim)
    pack_dtype: torch.dtype = CrModule.Constant(
        torch.uint8, validate=_validate_pack_dtype
    )

    # Eager tiling knobs (plain class attributes, not Constants):
    # _TILE_OUTPUTS bounds the output axis, _TILE_ROWS the row axis
    # (batch x positions for convolutions), and _TILE_BYTES shrinks the
    # row tile so a block never exceeds that many transient bytes.
    _TILE_OUTPUTS = 256
    _TILE_ROWS = 4096
    _TILE_BYTES = 1 << 26

    # Construction-time tail-mask buffer (annotation only: tells static
    # checkers what register_buffer provides at runtime).
    _cr_tail_mask_buf: Tensor

    # Packed weight buffer, declared per topology via ``Specs`` and
    # registered here at construction; annotation only, for checkers.
    weight: Tensor

    class Specs:
        y = CrModule.OutputSpec(differentiable=False)

    @classmethod
    def _cr_extra_init_params(cls) -> list[inspect.Parameter]:
        # One keyword-only weight kwarg per declared entry (None default);
        # crashing here would mean the collector above did not run.
        return [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[Tensor],
            )
            for name in cls._cr_weight_specs
        ]

    def __init__(self, **kwargs: Any) -> None:
        pending = {name: kwargs.pop(name, None) for name in self._cr_weight_specs}
        super().__init__(**kwargs)
        # Init-time split: loop-invariant integers live here, not in
        # _step. Buffers move with .to() like any buffer, and stay valid
        # because Constants are construction-time by contract (never
        # mutated post-construction).
        self.register_buffer(
            "_cr_tail_mask_buf",
            self._cr_tail_mask(torch.device("cpu")),
            persistent=True,
        )

        # Spatial specs normalize to rank-tuples here from their own
        # declared minima — no registry, no init method. Topologies
        # without spatial dims declare none, so rank is only ever read
        # when at least one exists.
        for name, spec in self._cr_constant_specs.items():
            if not isinstance(spec, SpatialSpec):
                continue

            spatial = getattr(self, "rank", 0) or 0
            setattr(
                self,
                name,
                normalize_spatial(getattr(self, name), spatial, name, spec.minimum),
            )

        vote_shapes: dict[str, tuple[int, ...]] = {}

        for name, spec in self._cr_weight_specs.items():
            shape = self._cr_resolve_weight_shape(spec)

            if spec.min_rank is not None and len(shape) < spec.min_rank:
                raise ValueError(
                    f"{type(self).__name__} weight {name!r} shape {shape} "
                    f"has rank {len(shape)}, below min_rank={spec.min_rank}; "
                    f"fix the Specs declaration"
                )

            if spec.max_rank is not None and len(shape) > spec.max_rank:
                raise ValueError(
                    f"{type(self).__name__} weight {name!r} shape {shape} "
                    f"has rank {len(shape)}, above max_rank={spec.max_rank}; "
                    f"fix the Specs declaration"
                )

            packed = self._cr_build_weight(pending[name], shape)
            self._cr_store_weight(name, packed, shape)
            vote_shapes[name] = shape

        if "weight" in vote_shapes:
            vote_shape = vote_shapes["weight"]
        elif len(vote_shapes) == 1:
            vote_shape = next(iter(vote_shapes.values()))
        else:
            raise ValueError(
                f"{type(self).__name__} declares {len(vote_shapes)} Specs "
                f"weights with no entry named 'weight'; the vote threshold "
                f"needs an unambiguous source (declare a Specs class first)"
            )

        votes = self.in_features * math.prod(vote_shape[2:], start=1)
        self._cr_n_votes = votes
        self._cr_vote_threshold = (votes + 1) // 2

    def _cr_resolve_weight_shape(self, spec: WeightSpec) -> tuple[int, ...]:
        """Resolve a Weight shape spec to concrete dims (tuples splat)."""
        out: list[int] = []

        for entry in spec.shape:
            if isinstance(entry, bool):
                raise TypeError(
                    f"{type(self).__name__} Weight shape entries must be "
                    f"ints or names, got {entry!r}"
                )

            if isinstance(entry, int):
                out.append(entry)
                continue

            if not isinstance(entry, str):
                raise TypeError(
                    f"{type(self).__name__} Weight shape entries must be "
                    f"ints or names, got {entry!r}"
                )

            value: Any = getattr(self, entry, None)

            if value is None and entry == "in_words":
                value = self._cr_in_words()

            if isinstance(value, bool):
                raise TypeError(
                    f"{type(self).__name__} Weight shape name {entry!r} "
                    f"resolved to a bool"
                )

            if isinstance(value, int):
                out.append(value)
            elif isinstance(value, (tuple, list)) and all(
                type(v) is int for v in value
            ):
                out.extend(value)
            else:
                raise ValueError(
                    f"{type(self).__name__} Weight shape name {entry!r} "
                    f"must resolve to an int or tuple of ints, got {value!r}"
                )

        return tuple(out)

    # --- subclass contract ---
    #
    # Weight plumbing (signature kwargs, 0/1 validation, packing,
    # canonical masking, buffer registration, threshold) lives here. A new
    # layer declares Specs plus _step, nothing else.

    def _cr_build_weight(
        self, weight: Optional[Tensor], shape: tuple[int, ...]
    ) -> Tensor:
        """Packed / unpacked-0/1 / None -> canonical packed buffer."""
        if weight is None:
            # Random bits, not zeros: all-zero weights make every channel
            # vote identically (a dead, symmetric network). Unpacked shape
            # is channels x taps; packing canonicalizes the tails.
            # Follows global RNG like any other init (seed for repro).
            unpacked = torch.randint(
                0,
                2,
                (shape[0], self.in_features, *shape[2:]),
                dtype=self.pack_dtype,
            )
            return self._cr_pack_channels(unpacked)

        if not isinstance(weight, Tensor):
            raise TypeError(
                f"{type(self).__name__} weight must be a tensor, "
                f"got {type(weight).__name__}"
            )

        if len(tuple(weight.shape)) != len(shape):
            raise ValueError(
                f"{type(self).__name__} weight must be rank {len(shape)}, "
                f"got shape {tuple(weight.shape)}"
            )

        if tuple(weight.shape) == shape and weight.dtype == self.pack_dtype:
            return weight.detach().clone()

        unpacked = (shape[0], self.in_features, *shape[2:])

        if tuple(weight.shape) == tuple(unpacked):
            if bool(((weight != 0) & (weight != 1)).any()):
                raise ValueError(
                    f"{type(self).__name__} unpacked weight must hold only "
                    f"0/1 values"
                )

            return self._cr_pack_channels(weight)

        raise ValueError(
            f"{type(self).__name__} weight must be packed {shape} "
            f"({self.pack_dtype}) or unpacked 0/1 {tuple(unpacked)}, got "
            f"{tuple(weight.shape)} {weight.dtype}"
        )

    # Word geometry (derived; plain methods, not Constants).

    def _cr_word_bits(self) -> int:
        return torch.iinfo(self.pack_dtype).bits

    def _cr_in_words(self) -> int:
        bits = self._cr_word_bits()
        return (self.in_features + bits - 1) // bits

    def _cr_tail_mask(self, device: torch.device) -> Tensor:
        """
        ``(in_words,)`` mask with 1s on valid bit positions (LSB-first),
        so padding bits in the last word never inflate agreement counts.
        """
        bits = self._cr_word_bits()
        words = self._cr_in_words()
        valid = self.in_features % bits or bits
        mask = torch.full(
            (words,), (1 << bits) - 1, dtype=self.pack_dtype, device=device
        )
        mask[-1] = (1 << valid) - 1
        return mask

    def _cr_pack_bits(self, bits_in: Tensor) -> Tensor:
        """``(..., F)`` {0,1}/bool -> ``(..., ceil(F / bits))`` packed LSB-first."""
        return _pack_bits_last(
            bits_in, self._cr_word_bits(), self.pack_dtype
        )

    def _cr_pack_channels(self, bits_in: Tensor) -> Tensor:
        """
        ``(B, C, *S)`` {0,1}/bool -> ``(B, Cw, *S)`` packed LSB-first
        along the channel dim.
        """
        return _pack_dim(
            bits_in, 1, self._cr_word_bits(), self.pack_dtype
        )

    def _cr_xnor_agree(self, x_packed: Tensor, w_packed: Tensor) -> Tensor:
        """
        ``(B, Wp)``, ``(O, Wp)`` -> ``(B, O)`` agreement counts.

        Counted via the complement (``N - disagreements``): one fewer pass
        than flipping to XNOR first, bit-identical. Weights arrive
        pre-masked (canonicalized at construction); the input is masked here on
        its small ``(B, Wp)`` tensor against the construction-time buffer,
        so the ``(B, O, Wp)`` transient is never masked.
        """
        xm = x_packed & self._cr_tail_mask_buf
        dis = _tiled_disagreement(
            xm,
            w_packed,
            self._cr_word_bits(),
            self._TILE_OUTPUTS,
            self._TILE_ROWS,
            self._TILE_BYTES,
        )
        return self.in_features - dis

    def _cr_check_packed(self, x: Tensor) -> None:
        words = self._cr_in_words()

        if x.dtype != self.pack_dtype:
            raise TypeError(
                f"{type(self).__name__} expects packed {self.pack_dtype} "
                f"input, got {x.dtype}; binarize floats first "
                f"(see XBinarize)"
            )

        if x.shape[-1] != words:
            raise ValueError(
                f"{type(self).__name__} expects packed input with {words} "
                f"words ({self.in_features} features), got shape "
                f"{tuple(x.shape)}"
            )

    def tally(self, inputs: InputTensor) -> Tensor:
        """
        Raw agreement tallies, ``(B, O)`` int32: per-(sample, channel)
        count of bit positions where input and weight point the same way.

        ``forward`` thresholds these at ``ceil(in_features / 2)`` and
        repacks. Margins ``|tally - thr|`` measure vote confidence, and
        pivotality derives exactly: an agreeing bit can flip the verdict
        iff ``tally == thr``, a disagreeing bit iff ``tally == thr - 1``.
        """
        canonical = self._cr_canonicalize_inputs(inputs)
        x = canonical[0]
        self._cr_check_packed(x)
        return self._cr_xnor_agree(x, self.weight)

    def _cr_store_weight(
        self, name: str, packed: Tensor, shape: tuple[int, ...]
    ) -> None:
        tail = self._cr_tail_mask(packed.device).reshape(
            (1, shape[1]) + (1,) * (len(shape) - 2)
        )
        self.register_buffer(name, packed & tail, persistent=True)

    def set_weight(self, weight: Tensor, name: str = "weight") -> "XBModule":
        """
        Inject externally-binarized weights (STE harness): same
        validation and canonicalization as construction (packed shape,
        unpacked 0/1, tail masking). Plain attribute assignment would
        skip all of that. Rejects None (construction already covers
        fresh initialization).
        """
        if weight is None:
            raise ValueError(
                f"{type(self).__name__}.set_weight needs a tensor, got None"
            )

        spec = self._cr_weight_specs.get(name)

        if spec is None:
            raise ValueError(
                f"{type(self).__name__} declares no weight {name!r}; "
                f"known: {sorted(self._cr_weight_specs)}"
            )

        shape = self._cr_resolve_weight_shape(spec)
        packed = self._cr_build_weight(weight, shape)
        self._cr_store_weight(name, packed, shape)
        return self

    # Device support: uint gating is construction-time (validate); kernel
    # support per device is probed lazily at the eager entries so the
    # compiled scan stays pure tensor math.

    def _cr_check_word_support(self, inputs: InputTensor) -> None:
        _check_word_support(self.pack_dtype, inputs, type(self).__name__)

    def forward(
        self,
        inputs: InputTensor,
        *state: Tensor,
    ) -> Tensor | StepOutput:
        self._cr_check_word_support(inputs)
        return super().forward(inputs, *state)

    def forward_sequence(
        self,
        x_seq: InputTensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        self._cr_check_word_support(x_seq)
        return super().forward_sequence(x_seq, state)

    def compile_sequence_scan(self, **kwargs: Any) -> "XBModule":
        # Base declares no weight buffer (layout is the subclass's); skip
        # the probe when there is nothing to probe yet.
        self._cr_check_word_support(getattr(self, "weight", ()))
        return super().compile_sequence_scan(**kwargs)
