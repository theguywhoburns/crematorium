from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from . import CrModule

Tensor: TypeAlias = torch.Tensor
StepOutput: TypeAlias = tuple[Tensor, ...]

InputTensor: TypeAlias = Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor]

Constraint: TypeAlias = Callable[[Tensor], Tensor]


# Constraints


def identity(t: Tensor) -> Tensor:
    """No constraint."""
    return t


def clamp_unit_interval(t: Tensor) -> Tensor:
    """Clamp to [0, 1]."""
    return torch.clamp(t, 0.0, 1.0)


def clamp_positive(t: Tensor) -> Tensor:
    """Clamp to be strictly positive."""
    return torch.clamp(t, min=1e-6)


def _cr_floating_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    State tensors should generally be floating point.
    If an integer dtype is requested, promote to default float dtype.
    """
    return dtype if dtype.is_floating_point else torch.get_default_dtype()


# Declarative parameter / state specs


@dataclass(frozen=True)
class ParamSpec:
    default: Any = None
    learnable: bool = False
    force_learn: bool = False
    constraint: Constraint = identity
    # Marks a param whose drift invalidates a frozen explicit spike_grad
    # (e.g. MCN's tau_L feeding an atan_surrogate beta): SnnModule raises
    # at construction for explicit spike_grad + learnable marked param.
    frozen_surrogate: bool = False
    dtype: Any = None


T = TypeVar("T")


@overload
def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    frozen_surrogate: bool = False,
    dtype: None = None,
) -> Any: ...


@overload
def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    frozen_surrogate: bool = False,
    dtype: type[T],
) -> T: ...


@overload
def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    frozen_surrogate: bool = False,
    dtype: Any = None,
) -> Any: ...


def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    frozen_surrogate: bool = False,
    dtype: Any = None,
) -> Any:
    """
    Declarative parameter field.

    With ``dtype=`` the call is typed as that Python type, so writing

        beta = CrModule.Param(0.9, dtype=float)

    makes the assigned attribute a ``float`` for static type checkers.
    Returns Any otherwise.

    Runtime note: at runtime the attribute is an ``nn.Parameter`` (a tensor
    scalar), not a Python ``float``. The ``dtype=`` argument exists so user
    code like ``self.beta * x`` type-checks against plain number literals;
    the generated ``__signature__`` advertises ``float | Tensor`` to keep
    help() honest about this.

    Constraints apply only to learnable parameters: a fixed (non-learnable)
    param is used raw in ``constrain(name)`` / resets, while a learnable one has
    its constraint applied on the hot path.

    ``frozen_surrogate`` marks a param whose drift invalidates a frozen
    explicit ``spike_grad`` (e.g. a surrogate beta tied to the param):
    ``SnnModule`` raises at construction for explicit ``spike_grad`` plus a
    learnable marked param.
    """
    return ParamSpec(
        default=default,
        learnable=learnable,
        force_learn=force_learn,
        constraint=constraint,
        frozen_surrogate=frozen_surrogate,
        dtype=dtype,
    )


@dataclass(frozen=True)
class ConstantSpec:
    """
    Declares a non-learnable hyperparameter on the module.

    Unlike ``Param``, a constant is never registered as an ``nn.Parameter``;
    it is exposed as a plain attribute and constructor kwarg.

    ``overridable`` controls per-instance constructor kwargs (subclass
    redeclaration of the default is always allowed):
      - ``True`` (default when unresolved): the constant is a constructor
        kwarg and appears in ``__signature__``.
      - ``False``: passing the kwarg raises ``TypeError`` and the name is
        hidden from ``__signature__``. The value is fixed to the default
        (a subclass may still redeclare a new default).
      - ``None``: inherit the nearest ancestor's resolved value walking
        base-first; ``True`` if no ancestor sets it. Reopening a locked
        (``False``) ancestor with ``True`` raises ``TypeError`` at class
        creation (monotonic lock).
    """

    default: Any = None
    validate: Optional[Callable[[Any], None]] = None
    dtype: Any = None
    overridable: bool | None = None


@overload
def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: None = None,
    overridable: bool | None = None,
) -> Any: ...


@overload
def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: type[T],
    overridable: bool | None = None,
) -> T: ...


@overload
def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: Any = None,
    overridable: bool | None = None,
) -> Any: ...


def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: Any = None,
    overridable: bool | None = None,
) -> Any:
    """
    Declarative non-learnable hyperparameter field.

    With ``dtype=`` the call is typed as that Python type, so writing

        dt = CrModule.Constant(0.01, dtype=float)

    makes the assigned attribute a ``float`` for static type checkers.
    Returns Any otherwise.

    ``validate``, if given, must raise ``ValueError`` on an invalid value; the
    framework re-raises with the module and field name attached.

    ``overridable=False`` pins the value to its default: passing it as a
    constructor kwarg raises ``TypeError`` and it is hidden from
    ``__signature__``. ``None`` (default) inherits the nearest ancestor's
    value, defaulting to ``True``.
    """
    return ConstantSpec(
        default=default,
        validate=validate,
        dtype=dtype,
        overridable=overridable,
    )


@dataclass(frozen=True)
class InputSpec:
    """
    Declares a named step input on a module.

    Inputs are declared positionally in a nested ``Inputs`` class:

        class Inputs:
            x: Tensor
            inh: Tensor

    The first declared input (or the one marked ``primary=True``) drives
    default state shapes (``StateSpec(shape="input")``) and hidden-mode
    device/dtype. ``dtype`` is stored as validation metadata only; it is not
    used to cast inputs.
    """

    primary: bool = False
    dtype: Any = None


@overload
def Input(*, primary: bool = False, dtype: None = None) -> Any: ...


@overload
def Input(*, primary: bool = False, dtype: type[T]) -> T: ...


def Input(*, primary: bool = False, dtype: Any = None) -> Any:
    """
    Declarative named-input field.

    With ``dtype=`` the call is typed as that Python type, so writing

        x = CrModule.Input(primary=True, dtype=float)

    makes the assigned attribute a ``float`` for static type checkers. Returns
    ``Any`` otherwise.
    """
    return InputSpec(primary=primary, dtype=dtype)


@dataclass(frozen=True)
class OutputSpec:
    """
    Declares an output tensor returned by `_step`.

    Outputs are not passed back into `_step` as recurrent state.
    ``differentiable=False`` detaches the tensor only when it is stored
    into a hidden-mode buffer; returned values keep the autograd graph
    so surrogate gradients can flow through them.
    """

    default: float | Callable[[nn.Module], float] = 0.0
    differentiable: bool = True


@dataclass(frozen=True)
class StateSpec:
    """
    Declares a recurrent state tensor passed into/out of `_step`.

    `shape` controls the state buffer's shape:
      - ``"input"`` (default) / ``None``: same shape as the primary input, so a
        (B, F) primary input yields a (B, F) state.
      - a string matching an input name: that input's shape. Useful for
        multi-input modules where a state should follow a non-primary input:

            v = CrModule.StateSpec(shape="inh")

      - an explicit tuple: decouples state shape from input shape:

            mem = CrModule.StateSpec(shape=(F,))   # per-feature state

    `None` behaves identically to "input" (the state follows the primary input
    shape); there is no scalar-shaped state convention.
    """

    default: float | Callable[[nn.Module], float] = 0.0
    differentiable: bool = True
    shape: str | tuple[int, ...] | None = "input"
    extras: dict = field(default_factory=dict)

    def __init__(
        self,
        default: float | Callable[[nn.Module], float] = 0.0,
        differentiable: bool = True,
        shape: str | tuple[int, ...] | None = "input",
        **extras: Any,
    ) -> None:
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "differentiable", differentiable)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "extras", extras)


Spec = Union[OutputSpec, StateSpec]

_PKModuleT = TypeVar("_PKModuleT", bound="type[CrModule]")


def extend_specs(**extensions: Callable[..., Any]):
    """
    Decorate a CrModule subclass with StateSpec extra handlers.

    Each extra key declared on a StateSpec (e.g. ``StateSpec(reset=...)``) is
    dispatched at construction time to the matching handler callable:

        @extend_specs(reset=ResetHandler)
        class SnnModule(CrModule):
            ...

    Handlers are called as ``handler(module, state_index, spec, value)``.
    """

    def decorator(cls: _PKModuleT) -> _PKModuleT:
        existing = dict(getattr(cls, "_cr_spec_extensions", {}))
        existing.update(extensions)
        cls._cr_spec_extensions = existing
        return cls

    return decorator
