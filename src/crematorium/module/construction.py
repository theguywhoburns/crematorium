from __future__ import annotations

import keyword
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .specs import Constraint, identity

if TYPE_CHECKING:
    from . import CrModule


def build_params(
    module: CrModule,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[Constraint]]:
    """
    Pop declared param/constant kwargs and register parameters.

    ``kwargs`` is consumed in place: every key matched to a ``ParamSpec`` or
    ``ConstantSpec`` is removed. Returns the leftover kwargs (unmatched keys)
    and the list of per-parameter constraint functions in declaration order.
    """
    constraint_fns: list[Constraint] = []

    _UNSET = object()

    for name, spec in module._cr_param_specs.items():
        value = kwargs.pop(name, spec.default)

        force_learn_kwarg = kwargs.pop(
            f"force_learn_{name}",
            _UNSET,
        )

        learnable_kwarg = kwargs.pop(
            f"learnable_{name}",
            _UNSET,
        )

        constraint = kwargs.pop(
            f"{name}_constraint",
            spec.constraint,
        )

        # force_learn=True means always learnable. An explicit
        # force_learn_<name>=True wins over everything; an explicit
        # learnable_<name>=False overrides spec-level force_learn.
        force_learn = (
            bool(force_learn_kwarg)
            if force_learn_kwarg is not _UNSET
            else spec.force_learn
        )

        if learnable_kwarg is not _UNSET:
            learnable = bool(learnable_kwarg)
            if force_learn_kwarg is not _UNSET and bool(force_learn_kwarg):
                learnable = True
        else:
            learnable = force_learn or spec.learnable

        if value is None:
            raise ValueError(
                f"{type(module).__name__} parameter {name!r} has no value or default"
            )

        tensor = torch.as_tensor(
            value,
            dtype=spec.dtype if isinstance(spec.dtype, torch.dtype) else None,
        )

        if not tensor.is_floating_point():
            tensor = tensor.to(torch.get_default_dtype())

        if isinstance(value, torch.Tensor):
            # Never alias caller-owned storage: later mutation of the
            # source tensor must not move the parameter.
            tensor = tensor.detach().clone()

        module.register_parameter(
            name,
            nn.Parameter(
                tensor,
                requires_grad=learnable,
            ),
        )

        # Fixed parameters skip constraint work entirely.
        constraint_fns.append(constraint if learnable else identity)

    for name, spec in module._cr_constant_specs.items():
        if name in kwargs and spec.overridable is False:
            raise TypeError(
                f"{type(module).__name__} constant {name!r} is not "
                f"overridable (overridable=False); it is fixed to its default"
            )

        value = kwargs.pop(name, spec.default)
        value_orig = value

        if spec.validate is not None:
            try:
                spec.validate(value)
            except ValueError as exc:
                raise ValueError(f"{type(module).__name__} {name}: {exc}") from None

        if isinstance(spec.dtype, type):
            try:
                value = spec.dtype(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{type(module).__name__} {name}: value {value_orig!r} "
                    f"cannot be stored as {spec.dtype.__name__}"
                ) from None
            if value != value_orig:
                raise ValueError(
                    f"{type(module).__name__} {name}: value {value_orig!r} "
                    f"cannot be stored as {spec.dtype.__name__}"
                )

        setattr(module, name, value)

    return kwargs, constraint_fns


def _cr_install_constrained(module: CrModule) -> None:
    """
    Freeze the constrained-parameter accessors on the module.

    Builds the per-instance ``name -> constraint`` map (``None`` where the
    constraint is identity, i.e. fixed params) backing ``constrain(name)``:
    one dict lookup plus one attribute lookup plus one constraint call per
    requested parameter, with no metadata resolution on the hot path.
    No ``exec`` and no cache: the map is built per module and keeps no
    references whose lifetime could interact with GC (the old ``id()``-keyed
    cache was safe only by accident).

    Constrained (non-identity) constraint callables are still exposed as
    ``_cr_constraint_{i}`` attributes because the SNN reset codegen
    (``ResetMixin._cr_constraint_expr``) references them.
    """
    param_names = tuple(module._cr_param_specs.keys())

    for name in param_names:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(
                f"Param name {name!r} must be a valid non-keyword Python identifier"
            )

    for i, constraint in enumerate(module._cr_constraints):
        if constraint is not identity:
            setattr(module, f"_cr_constraint_{i}", constraint)

    module._cr_param_constraint_map = {
        name: None if constraint is identity else constraint
        for name, constraint in zip(param_names, module._cr_constraints, strict=True)
    }


def remaining_kwargs_error(
    module: CrModule,
    kwargs: dict[str, Any],
) -> TypeError:
    """
    Build the leftover-kwarg TypeError with the exact message users expect.
    """
    return TypeError(
        f"{type(module).__name__} got unexpected keyword arguments: {sorted(kwargs)}"
    )
