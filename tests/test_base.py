from __future__ import annotations

import copy
import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

from crematorium import (
    CrModule,
    OutputSpec,
    Param,
    ParamSpec,
    StateSpec,
    clamp_positive,
    clamp_unit_interval,
    get_validation,
    identity,
    no_validation,
    set_sequence_scan_chunk,
    set_validation,
)
from crematorium.snn import SnnModule
from crematorium.snn.neurons.LIF import LIF

B, F = 4, 8
X = torch.randn(B, F)
T = 5
X_SEQ = torch.randn(T, B, F)


# ----------------------------------------------------------------------
# Shared test doubles (section 2 of the plan)
# ----------------------------------------------------------------------


class _Leaky(CrModule):
    """Single output (mem), single state (mem). Pure leaky integrator."""

    class Params:
        beta: float = CrModule.Param(
            0.5,
            constraint=clamp_unit_interval,
        )

    class Specs:
        out = CrModule.OutputSpec(differentiable=False)
        mem = CrModule.StateSpec()

    def _step(self, x, mem):
        beta = self.constrain("beta")
        mem = beta * mem + x
        return mem, mem  # (out, next_mem)


class _TwoOut(CrModule):
    """Two outputs + one state: multi-output path."""

    class Params:
        beta: float = CrModule.Param(0.5)

    class Specs:
        o1 = CrModule.OutputSpec()
        o2 = CrModule.OutputSpec(differentiable=False)
        mem = CrModule.StateSpec()

    def _step(self, x, mem):
        mem = self.beta * mem + x
        return mem, mem * 2.0, mem


class _WrongCount(CrModule):
    """_step returns 2 tensors but Specs declares 3 entries."""

    class Specs:
        o = CrModule.OutputSpec()
        s1 = CrModule.StateSpec()
        s2 = CrModule.StateSpec()

    def _step(self, x, s1, s2):
        return x, x  # wrong: only 2


class _NoTuple(CrModule):
    """_step returns a bare Tensor, not a tuple."""

    class Specs:
        o = CrModule.OutputSpec()
        s = CrModule.StateSpec()

    def _step(self, x, s):
        return x  # wrong: not a tuple


class _TypedInput(CrModule):
    """Primary input declared with dtype=float (non-structural dtype check)."""

    class Inputs:
        x: torch.Tensor = CrModule.Input(dtype=float)

    class Specs:
        o = CrModule.OutputSpec()
        s = CrModule.StateSpec()

    def _step(self, x, s):
        return x, s


class _FixedShapeHidden(CrModule):
    """StateSpec with an explicit tuple shape (hidden allocation honors it)."""

    class Specs:
        o = CrModule.OutputSpec(differentiable=False)
        mem = CrModule.StateSpec(shape=(B, 2 * F))

    def _step(self, x, mem):
        return torch.zeros_like(x), mem


class _NoneShape(CrModule):
    """StateSpec with shape=None (treated as "input", follows the input shape)."""

    class Specs:
        o = CrModule.OutputSpec(differentiable=False)
        mem = CrModule.StateSpec(shape=None)

    def _step(self, x, mem):
        return torch.zeros_like(x), mem


class _CallableDefault(CrModule):
    """Spec default that is a callable on the module."""

    class Specs:
        o = CrModule.OutputSpec(differentiable=False)
        mem = CrModule.StateSpec(default=lambda m: 0.25)

    def _step(self, x, mem):
        return torch.zeros_like(x), mem


class _NoParams(CrModule):
    """Empty Params + one output/state."""

    class Specs:
        o = CrModule.OutputSpec()
        s = CrModule.StateSpec()

    def _step(self, x, s):
        return x, x


class _ForceLearn(CrModule):
    """Spec-level force_learn=True."""

    class Params:
        beta: float = CrModule.Param(0.5, force_learn=True)

    class Specs:
        o = CrModule.OutputSpec()
        s = CrModule.StateSpec()

    def _step(self, x, s):
        return x, x


class _NoneDefault(CrModule):
    """Param whose default is None: construction must raise."""

    class Params:
        a: float = CrModule.Param(None)

    class Specs:
        o = CrModule.OutputSpec()
        s = CrModule.StateSpec()

    def _step(self, x, s):
        return x, x


class _PlainRNN(CrModule):
    """Generic non-SNN RNN cell: pure state threading, no spike semantics."""

    class Params:
        w: float = CrModule.Param(0.5)

    class Specs:
        out = CrModule.OutputSpec()
        h = CrModule.StateSpec()

    def _step(self, x, h):
        h = torch.tanh(self.w * h + x)
        return h, h


# ----------------------------------------------------------------------
# A. Constraints (pure functions)
# ----------------------------------------------------------------------


def test_identity_is_noop():
    t = torch.randn(3, 4)
    assert identity(t) is t
    assert torch.equal(identity(t), t)


def test_clamp_unit_interval_clamps():
    x = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.0])
    out = clamp_unit_interval(x)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert out.dtype == x.dtype
    assert out.shape == x.shape


def test_clamp_positive_floor():
    x = torch.tensor([0.0, -5.0, 1e-6, 2.0])
    out = clamp_positive(x)
    assert torch.equal(out, torch.tensor([1e-6, 1e-6, 1e-6, 2.0]))
    assert out.dtype == x.dtype


def test_constraints_grad_compatible():
    t = torch.randn(4, requires_grad=True)
    for fn in (identity, clamp_unit_interval, clamp_positive):
        t.grad = None
        fn(t).sum().backward()
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


def test_safe_exp_matches_exp_where_finite():
    for dtype in (torch.float32, torch.float64):
        x = torch.linspace(-100.0, 80.0, 200, dtype=dtype)
        assert torch.equal(CrModule.safe_exp(x), x.exp())


def test_safe_exp_stays_finite_at_extremes():
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([-1e6, 0.0, 1e6, 1e30], dtype=dtype)
        out = CrModule.safe_exp(x)
        assert torch.isfinite(out).all()
        assert torch.equal(out[1], torch.tensor(1.0, dtype=dtype))
        max_arg = torch.log(torch.tensor(torch.finfo(dtype).max, dtype=dtype)) - 1
        assert out[-1].item() == torch.exp(max_arg).item()


def test_safe_exp_grad_stays_finite():
    x = torch.tensor([-1e6, 1e6], requires_grad=True)
    CrModule.safe_exp(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ----------------------------------------------------------------------
# B. Param / Spec metadata
# ----------------------------------------------------------------------


def test_param_spec_field_defaults():
    p = Param()
    assert isinstance(p, ParamSpec)
    assert p.default is None
    assert p.learnable is False
    assert p.force_learn is False
    assert p.constraint is identity


def test_param_is_frozen():
    p = ParamSpec()
    with pytest.raises(FrozenInstanceError):
        p.default = 1.0


def test_output_state_spec_defaults():
    o = OutputSpec()
    assert o.default == 0.0
    assert o.differentiable is True
    s = StateSpec()
    assert s.default == 0.0
    assert s.differentiable is True
    assert s.shape == "input"


def test_module_metadata_collected():
    m = _Leaky()
    assert set(m._cr_param_specs) == {"beta"}
    assert m._cr_output_names == ("out",)
    assert m._cr_state_names == ("mem",)
    assert len(m._cr_output_specs) == 1
    assert len(m._cr_state_specs) == 1
    names = [name for name, _ in m._cr_spec_entries]
    assert names == ["out", "mem"]


def test_mro_param_merge():
    class _Base(CrModule):
        class Params:
            beta: float = CrModule.Param(0.5)

        class Specs:
            o = CrModule.OutputSpec()
            s = CrModule.StateSpec()

        def _step(self, x, s):
            return x, x

    class _Child(_Base):
        class Params:
            beta: float = CrModule.Param(0.9)

    c = _Child()
    assert "beta" in c._cr_param_specs
    assert c.beta.item() == pytest.approx(0.9)


def test_mro_spec_merge():
    class _Base(CrModule):
        class Specs:
            o = CrModule.OutputSpec()
            s = CrModule.StateSpec()

        def _step(self, x, s):
            return x, x

    class _Child(_Base):
        class Specs:
            s2 = CrModule.StateSpec()

        def _step(self, x, s, s2):
            return x, s, s2

    c = _Child()
    assert c._cr_output_names == ("o",)
    assert c._cr_state_names == ("s", "s2")
    names = [name for name, _ in c._cr_spec_entries]
    assert names == ["o", "s", "s2"]


def test_param_value_dtype_promotion():
    m = _Leaky(beta=1)
    assert m.beta.dtype == torch.get_default_dtype()

    m32 = _Leaky(beta=torch.tensor(0.5, dtype=torch.float32))
    assert m32.beta.dtype == torch.float32

    m64 = _Leaky(beta=torch.tensor(0.5, dtype=torch.float64))
    assert m64.beta.dtype == torch.float64


def test_param_missing_value_raises():
    with pytest.raises(ValueError, match="no value or default"):
        _NoneDefault()


def test_unexpected_kwarg_raises():
    with pytest.raises(TypeError, match="unexpected keyword"):
        _Leaky(bogus=1)
    with pytest.raises(TypeError, match="bogus"):
        _Leaky(bogus=1)


def test_per_param_kwarg_overrides():
    m = _Leaky(beta=2.0, learnable_beta=True, beta_constraint=clamp_positive)
    assert m.beta.requires_grad is True
    assert m.constrain("beta").item() == 2.0

    m_unit = _Leaky(beta=2.0, learnable_beta=True)
    assert m_unit.constrain("beta").item() == 1.0


def test_force_learn_forces_learnable():
    m = _ForceLearn()
    assert m.beta.requires_grad is True

    m2 = _Leaky(force_learn_beta=True, learnable_beta=False)
    assert m2.beta.requires_grad is True


def test_learnable_false_overrides_spec_force_learn():
    m = _ForceLearn(learnable_beta=False)
    assert m.beta.requires_grad is False


def test_force_learn_false_kwarg_cancels_spec():
    m = _ForceLearn(force_learn_beta=False, learnable_beta=False)
    assert m.beta.requires_grad is False

    m2 = _ForceLearn(force_learn_beta=False)
    assert m2.beta.requires_grad is False


def test_invalid_param_name_raises():
    def _step(self, x, s):
        return x, x

    P = type("Params", (), {})
    setattr(P, "for", Param(1.0))
    S = type(
        "Specs",
        (),
        {
            "o": CrModule.OutputSpec(),
            "s": CrModule.StateSpec(),
        },
    )
    # A keyword param name is rejected while the runtime signature is built
    # (at class creation, before __init__/`_install_constrained` runs).
    with pytest.raises(ValueError, match="not a valid parameter name"):
        type("M", (CrModule,), {"Params": P, "Specs": S, "_step": _step})


def test_constrain_resolves_by_name():
    m = _Leaky()
    assert m.constrain("beta") is m.beta

    lif = LIF()
    assert lif.constrain("beta") is lif.beta
    assert lif.constrain("threshold") is lif.threshold

    with pytest.raises(KeyError, match="unknown Param"):
        m.constrain("bogus")


def test_constraint_applied_only_when_learnable():
    m = _Leaky(beta=2.0)
    assert m.constrain("beta").item() == 2.0

    m_learn = _Leaky(beta=2.0, learnable_beta=True)
    assert m_learn.constrain("beta").item() == 1.0


def test_constraint_policy_full_matrix():
    # Fixed param out of range -> used raw.
    fixed = _Leaky(beta=2.0)
    assert fixed.constrain("beta").item() == 2.0

    # Learnable param out of range -> clamped (unit interval).
    learn = _Leaky(beta=2.0, learnable_beta=True)
    assert learn.constrain("beta").item() == 1.0

    # Custom per-param constraint override replaces the spec constraint.
    custom = _Leaky(beta=-5.0, learnable_beta=True, beta_constraint=clamp_positive)
    assert custom.constrain("beta").item() == pytest.approx(1e-6)

    # Custom constraint is only used while learnable; fixed stays raw.
    fixed_custom = _Leaky(beta=-5.0, beta_constraint=clamp_positive)
    assert fixed_custom.constrain("beta").item() == -5.0


def test_constrained_dict_empty_when_no_params():
    m = _NoParams()
    assert m.constrained_dict() == {}


# ----------------------------------------------------------------------
# C. Validation toggle (global + per-module)
# ----------------------------------------------------------------------


def test_global_validation_default():
    set_validation(True)
    assert get_validation() is True


def test_set_validation_coerces_to_bool():
    set_validation(0)
    assert get_validation() is False
    set_validation(1)
    assert get_validation() is True
    set_validation("x")
    assert get_validation() is True


def test_no_validation_context():
    prev = get_validation()
    with no_validation():
        assert get_validation() is False
        with no_validation():
            assert get_validation() is False
    assert get_validation() is prev


def test_no_validation_restores_on_exception():
    prev = get_validation()
    with pytest.raises(RuntimeError):
        with no_validation():
            assert get_validation() is False
            raise RuntimeError("boom")
    assert get_validation() is prev


def test_module_validate_follows_global():
    set_validation(False)
    m = _Leaky()
    assert m.validate is False

    set_validation(True)
    m2 = _Leaky()
    assert m2.validate is True


def test_constructor_validate_override_wins():
    m = _Leaky(validate=False)
    with no_validation():
        assert m.validate is False
    set_validation(True)
    assert m.validate is False

    m2 = _Leaky(validate=True)
    with no_validation():
        assert m2.validate is True


def test_validate_setter_pins_module():
    m = _Leaky()
    m.validate = False
    set_validation(True)
    assert m.validate is False
    with no_validation():
        assert m.validate is False


def test_validate_setter_bool_coercion():
    m = _Leaky()
    m.validate = 0
    assert m.validate is False
    m.validate = 1
    assert m.validate is True


def test_validate_gates_state_count_check():
    x = torch.randn(B, F)
    state = _Leaky().initial_state((B, F))

    m = _Leaky(validate=True)
    with pytest.raises(ValueError, match="expects 1 state"):
        m(x, state[0], state[0])

    m2 = _Leaky(validate=False)
    out = m2(x, state[0])
    assert isinstance(out, tuple) and len(out) == 2


def test_step_output_count_checked_even_without_validation():
    # Structural output-arity check is O(1) and stays on in fast mode: a
    # _step returning the wrong count fails loudly instead of handing the
    # caller silently garbled output when validate=False.
    x = torch.randn(B, F)
    s1 = torch.randn(B, F)
    s2 = torch.randn(B, F)

    m = _WrongCount(validate=True)
    with pytest.raises(ValueError, match="returned 2 tensors, expected 3"):
        m(x, s1, s2)

    m2 = _WrongCount(validate=False)
    with pytest.raises(ValueError, match="returned 2 tensors, expected 3"):
        m2(x, s1, s2)


def test_step_output_type_checked_even_without_validation():
    # Structural tuple check stays on in fast mode: a bare-tensor _step
    # return is a caller-visible contract break, not an optimization.
    x = torch.randn(B, F)
    s = torch.randn(B, F)

    m = _NoTuple(validate=True)
    with pytest.raises(TypeError, match="must return a tuple of tensors"):
        m(x, s)

    m2 = _NoTuple(validate=False)
    with pytest.raises(TypeError, match="must return a tuple of tensors"):
        m2(x, s)


def test_validate_setter_dynamic_toggle():
    # The dynamic toggle still gates the non-structural per-input dtype
    # checks: an integer tensor on a float-declared input raises only while
    # validation is on.
    m = _TypedInput(validate=False)
    x_long = torch.zeros(B, F, dtype=torch.long)
    s = torch.randn(B, F)

    out, *_ = m(x_long, s)
    assert out.shape == (B, F)

    m.validate = True
    with pytest.raises(TypeError, match="declared with dtype="):
        m(x_long, s)


# ----------------------------------------------------------------------
# D. Hidden vs explicit step paths & outputs
# ----------------------------------------------------------------------


def test_hidden_forward_single_output_returns_tensor():
    m = _Leaky(init_hidden=True)
    out = m(torch.randn(B, F))
    assert isinstance(out, torch.Tensor)


def test_hidden_forward_multi_output_returns_tuple():
    m = _TwoOut(init_hidden=True)
    out = m(torch.randn(B, F))
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_hidden_forward_rejects_explicit_state():
    m = _Leaky(init_hidden=True)
    state = torch.randn(B, F)
    with pytest.raises(
        ValueError,
        match="init_hidden=True mode; do not pass state explicitly",
    ):
        m(torch.randn(B, F), state)


def test_explicit_forward_returns_full_tuple():
    m = _Leaky(init_hidden=False)
    state = m.initial_state((B, F))
    out = m(torch.randn(B, F), *state)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_hidden_explicit_equivalence():
    torch.manual_seed(0)
    hidden = _Leaky(init_hidden=True)
    explicit = _Leaky(init_hidden=False)
    state = explicit.initial_state((B, F))

    for _ in range(T):
        x = torch.randn(B, F)
        h_out = hidden(x)
        e_out = explicit(x, *state)
        assert torch.equal(h_out, e_out[0])
        state = e_out[1:]
        assert torch.allclose(hidden._buffers["mem"], state[0], atol=1e-6)


def test_hidden_buffer_lazy_allocation():
    m = _Leaky(init_hidden=True)
    assert m._cr_allocated is False
    assert "mem" not in m._buffers
    m(torch.randn(B, F))
    assert m._cr_allocated is True
    assert "mem" in m._buffers


def test_hidden_buffers_non_persistent():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))
    assert "mem" not in m.state_dict().keys()
    assert m.mem is m._buffers["mem"]


def test_hidden_dtype_follows_input():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F, dtype=torch.float64))
    assert m._buffers["mem"].dtype == torch.float64

    m2 = _Leaky(init_hidden=True)
    m2(torch.randint(0, 2, (B, F), dtype=torch.int64))
    assert m2._buffers["mem"].dtype == torch.get_default_dtype()


def test_hidden_defaults_fill_spec_values():
    m = _CallableDefault(init_hidden=True)
    m(torch.randn(B, F))
    assert torch.allclose(m._buffers["mem"], torch.full((B, F), 0.25))


def test_hidden_alloc_honors_explicit_shape():
    m = _FixedShapeHidden(init_hidden=True)
    m(torch.randn(B, F))
    assert m._buffers["mem"].shape == (B, 2 * F)
    assert m._buffers["o"].shape == (B, F)


def test_hidden_explicit_shape_allows_repeated_calls():
    # An explicit StateSpec shape is decoupled from the input shape, so the
    # hidden shape guard must not reject a repeated same-shaped call.
    m = _FixedShapeHidden(init_hidden=True)
    m(torch.randn(B, F))
    m(torch.randn(B, F))
    assert m._buffers["mem"].shape == (B, 2 * F)


def test_hidden_state_persists_across_steps():
    m = _Leaky(init_hidden=True)
    out1 = m(torch.ones(B, F))
    out2 = m(torch.zeros(B, F))
    assert not torch.equal(out1, out2)


def test_hidden_non_differentiable_output_detached():
    m = _TwoOut(init_hidden=True)
    x = torch.randn(B, F, requires_grad=True)
    m(x)
    assert m._buffers["o2"].requires_grad is False
    assert m._buffers["mem"].requires_grad is True


def test_explicit_outputs_keep_graph():
    m = _TwoOut(init_hidden=False)
    state = m.initial_state((B, F))
    x = torch.randn(B, F, requires_grad=True)
    o1, o2, mem = m(x, *state)
    assert o1.grad_fn is not None
    assert o2.grad_fn is not None
    assert mem.grad_fn is not None


def test_step_state_contract_single():
    m = _Leaky(init_hidden=False)
    state = m.initial_state((B, F))
    x = torch.randn(B, F)

    spk, next_state = m.step_state(x, state)
    assert isinstance(spk, torch.Tensor)
    assert isinstance(next_state, tuple) and len(next_state) == 1

    fwd = m.forward(x, *state)
    assert isinstance(fwd, tuple)
    assert torch.allclose(spk, fwd[0])
    assert torch.allclose(next_state[0], fwd[1])


def test_step_state_contract_multi():
    m = _TwoOut(init_hidden=False)
    state = m.initial_state((B, F))
    x = torch.randn(B, F)

    outputs, next_state = m.step_state(x, state)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert isinstance(next_state, tuple) and len(next_state) == 1
    assert torch.allclose(outputs[1], outputs[0] * 2.0)
    assert torch.allclose(next_state[0], outputs[0])


def test_step_state_requires_explicit_mode():
    m = _Leaky(init_hidden=True)
    with pytest.raises(ValueError, match="step_state requires init_hidden=False"):
        m.step_state(torch.randn(B, F), ())


def test_step_alias_equals_step_state():
    m = _Leaky(init_hidden=False)
    state = m.initial_state((B, F))
    x = torch.randn(B, F)

    spk_a, next_a = m.step(x, state)
    spk_b, next_b = m.step_state(x, state)
    assert torch.allclose(spk_a, spk_b)
    assert torch.allclose(next_a[0], next_b[0])


def test_step_state_matches_manual_loop():
    m = _Leaky(init_hidden=False)
    state = m.initial_state((B, F))
    x = torch.randn(B, F)

    _, next_state = m.step_state(x, state)
    mem = state[0]
    mem = 0.5 * mem + x
    assert torch.allclose(next_state[0], mem, atol=1e-6)


# ----------------------------------------------------------------------
# E. State factories
# ----------------------------------------------------------------------


def test_initial_state_fills_defaults():
    m = _Leaky()
    state = m.initial_state((B, F))
    assert isinstance(state, tuple) and len(state) == 1
    assert state[0].shape == (B, F)
    default = m._cr_state_specs[0].default
    assert torch.allclose(state[0], torch.full((B, F), float(default)))


def test_initial_state_count_is_state_specs_only():
    m = _TwoOut()
    state = m.initial_state((B, F))
    assert len(state) == 1


def test_initial_state_dtype():
    m = _Leaky()
    s64 = m.initial_state((B, F), dtype=torch.float64)
    assert s64[0].dtype == torch.float64

    s_int = m.initial_state((B, F), dtype=torch.int64)
    assert s_int[0].dtype == torch.get_default_dtype()

    s_none = m.initial_state((B, F))
    assert s_none[0].dtype == torch.get_default_dtype()


def test_initial_state_device():
    m = _Leaky()
    cpu = m.initial_state((B, F), device=torch.device("cpu"))
    assert cpu[0].device.type == "cpu"

    if torch.cuda.is_available():
        cuda = m.initial_state((B, F), device=torch.device("cuda"))
        assert cuda[0].device.type == "cuda"


def test_initial_state_resolves_callable_default():
    m = _CallableDefault()
    state = m.initial_state((B, F))
    assert torch.allclose(state[0], torch.full((B, F), 0.25))


def test_zero_state_zeros():
    m = _TwoOut()
    state = m.zero_state((B, F))
    assert len(state) == 1
    assert state[0].shape == (B, F)
    assert torch.equal(state[0], torch.zeros(B, F))


def test_initial_state_like():
    m = _Leaky()
    x = torch.randn(B, F, dtype=torch.float64)
    state = m.initial_state_like(x)
    assert state[0].shape == x.shape
    assert state[0].dtype == torch.float64
    assert state[0].device == x.device

    reshaped = m.initial_state_like(x, batch_shape=(2, F))
    assert reshaped[0].shape == (2, F)


def test_initial_state_for_sequence():
    m = _Leaky()
    x_seq = torch.randn(T, B, F, dtype=torch.float64)
    state = m.initial_state_for_sequence(x_seq)
    assert state[0].shape == (B, F)
    assert state[0].dtype == torch.float64
    assert state[0].device == x_seq.device


def test_initial_state_for_sequence_too_few_dims():
    m = _Leaky()
    x = torch.randn(B, F)
    with pytest.raises(ValueError, match=r"expects \(time, batch, features\)"):
        m.initial_state_for_sequence(x)


def test_initial_state_honors_explicit_shape():
    m = _FixedShapeHidden()
    state = m.initial_state((B, F))
    assert state[0].shape == (B, 2 * F)


def test_initial_state_input_and_none_follow_input_shape():
    m_input = _Leaky()
    state = m_input.initial_state((B, F))
    assert state[0].shape == (B, F)

    m_none = _NoneShape()
    state = m_none.initial_state((B, F))
    assert state[0].shape == (B, F)


def test_zero_state_honors_explicit_shape():
    m = _FixedShapeHidden()
    state = m.zero_state((B, F))
    assert state[0].shape == (B, 2 * F)


def test_explicit_state_factories_match_hidden_allocation():
    m = _FixedShapeHidden(init_hidden=True)
    m(torch.randn(B, F))
    assert m._buffers["mem"].shape == (B, 2 * F)

    m2 = _FixedShapeHidden(init_hidden=False)
    init = m2.initial_state((B, F))
    zero = m2.zero_state((B, F))
    assert init[0].shape == (B, 2 * F)
    assert zero[0].shape == (B, 2 * F)

    x_seq = torch.randn(T, B, F)
    seq_init = m2.initial_state_for_sequence(x_seq)
    assert seq_init[0].shape == (B, 2 * F)


# ----------------------------------------------------------------------
# F. allocate_like / lazy hidden allocation
# ----------------------------------------------------------------------


def test_allocate_like_materializes_without_forward():
    m = _Leaky(init_hidden=True)
    x = torch.randn(B, F, dtype=torch.float64)
    m.allocate_like(x)
    assert m._cr_allocated is True
    assert "mem" in m._buffers
    assert m._buffers["mem"].shape == x.shape
    assert m._buffers["mem"].dtype == x.dtype
    assert m._buffers["mem"].device == x.device
    assert torch.allclose(m._buffers["mem"], torch.zeros(B, F, dtype=torch.float64))


def test_allocate_like_noop_when_explicit():
    m = _Leaky(init_hidden=False)
    ret = m.allocate_like(torch.randn(B, F))
    assert ret is m
    assert m._cr_allocated is False
    assert "mem" not in m._buffers


def test_allocate_like_idempotent_preserves_state():
    m = _Leaky(init_hidden=True)
    x = torch.randn(B, F)
    m.allocate_like(x)
    m(x)
    after_step = m._buffers["mem"].clone()
    m.allocate_like(x)
    assert torch.allclose(m._buffers["mem"], after_step)


def test_allocate_like_returns_self():
    m = _Leaky(init_hidden=True)
    assert m.allocate_like(torch.randn(B, F)) is m


def test_hidden_shape_change_raises():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))

    with pytest.raises(ValueError, match="shape"):
        m(torch.randn(B + 1, F))


def test_hidden_shape_change_sequence_raises():
    m = _Leaky(init_hidden=True)
    m.forward_sequence(torch.randn(T, B, F))

    with pytest.raises(ValueError, match="shape"):
        m.forward_sequence(torch.randn(T, B + 1, F))


def test_hidden_shape_change_validation_off_skips_friendly_error():
    # With validation disabled the friendly ValueError is skipped; the raw
    # tensor shape mismatch surfaces from inside the step math instead.
    m = _Leaky(init_hidden=True)
    m.fast_sequence_(compile_scan=False)
    m(torch.randn(B, F))

    with pytest.raises(RuntimeError, match="must match"):
        m(torch.randn(B + 1, F))


def test_hidden_shape_change_error_mentions_fixed_dims():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))
    with pytest.raises(ValueError, match="batch/feature dims must stay fixed"):
        m(torch.randn(B, F + 1))


# ----------------------------------------------------------------------
# G. forward_sequence
# ----------------------------------------------------------------------


def test_forward_sequence_hidden_single_shape():
    m = _Leaky(init_hidden=True)
    out = m.forward_sequence(torch.randn(T, B, F))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (T, B, F)


def test_forward_sequence_hidden_multi_tuple():
    m = _TwoOut(init_hidden=True)
    out = m.forward_sequence(torch.randn(T, B, F))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (T, B, F)
    assert out[1].shape == (T, B, F)


def test_forward_sequence_explicit_single():
    m = _Leaky(init_hidden=False)
    out = m.forward_sequence(torch.randn(T, B, F))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (T, B, F)
    assert out[1].shape == (B, F)


def test_forward_sequence_explicit_multi():
    m = _TwoOut(init_hidden=False)
    out = m.forward_sequence(torch.randn(T, B, F))
    assert isinstance(out, tuple) and len(out) == 3
    assert out[0].shape == (T, B, F)
    assert out[1].shape == (T, B, F)
    assert out[2].shape == (B, F)


def test_forward_sequence_explicit_default_state():
    torch.manual_seed(0)
    m = _Leaky(init_hidden=False)
    x_seq = torch.randn(T, B, F)
    seq, final = m.forward_sequence(x_seq)

    ref = _Leaky(init_hidden=False)
    state = ref.initial_state_for_sequence(x_seq)
    cur = state
    outs = []
    for t in range(T):
        spk, cur = ref.step_state(x_seq[t], cur)
        outs.append(spk)
    ref_seq = torch.stack(outs)
    assert torch.allclose(seq, ref_seq, atol=1e-6)
    assert torch.allclose(final, cur[0], atol=1e-6)


def test_forward_sequence_hidden_rejects_state():
    m = _Leaky(init_hidden=True)
    state = (torch.zeros(B, F),)
    with pytest.raises(ValueError, match="does not accept explicit state"):
        m.forward_sequence(torch.randn(T, B, F), state)


def test_forward_sequence_too_few_dims():
    m = _Leaky()
    with pytest.raises(ValueError, match=r"expects \(time, batch, features\)"):
        m.forward_sequence(torch.randn(B, F))


def test_forward_sequence_empty_raises():
    m = _Leaky()
    with pytest.raises(ValueError, match="at least one timestep"):
        m.forward_sequence(torch.empty(0, B, F))


def test_forward_sequence_T1():
    m = _Leaky(init_hidden=True)
    out = m.forward_sequence(torch.randn(1, B, F))
    assert out.shape == (1, B, F)


def test_forward_sequence_hidden_matches_explicit():
    torch.manual_seed(0)
    hidden = _Leaky(init_hidden=True)
    explicit = _Leaky(init_hidden=False)
    x_seq = torch.randn(T, B, F)
    h_seq = hidden.forward_sequence(x_seq)
    e_seq, final = explicit.forward_sequence(x_seq)
    assert torch.allclose(h_seq, e_seq, atol=1e-6)
    assert torch.allclose(hidden._buffers["mem"], final, atol=1e-6)


def test_forward_sequence_matches_step_loop_explicit():
    torch.manual_seed(0)
    m = _Leaky(init_hidden=False)
    x_seq = torch.randn(T, B, F)
    seq, final = m.forward_sequence(x_seq)

    ref = _Leaky(init_hidden=False)
    state = ref.initial_state((B, F))
    outs = []
    for t in range(T):
        spk, state = ref.step_state(x_seq[t], state)
        outs.append(spk)
    assert torch.equal(seq, torch.stack(outs))
    assert torch.allclose(final, state[0], atol=1e-6)


def test_forward_sequence_matches_step_loop_hidden():
    torch.manual_seed(0)
    m = _Leaky(init_hidden=True)
    x_seq = torch.randn(T, B, F)
    seq = m.forward_sequence(x_seq)

    ref = _Leaky(init_hidden=True)
    outs = []
    for t in range(T):
        outs.append(ref(x_seq[t]))
    assert torch.equal(seq, torch.stack(outs))


def test_forward_sequence_eager_chunked_path():
    torch.manual_seed(0)
    T20 = 20
    m = _Leaky(init_hidden=False)
    x_seq = torch.randn(T20, B, F)
    seq, final = m.forward_sequence(x_seq)

    ref = _Leaky(init_hidden=False)
    state = ref.initial_state((B, F))
    outs = []
    for t in range(T20):
        spk, state = ref.step_state(x_seq[t], state)
        outs.append(spk)
    assert torch.equal(seq, torch.stack(outs))
    assert torch.allclose(final, state[0], atol=1e-6)


def test_set_sequence_scan_chunk_preserves_results():
    from crematorium.module import _SEQUENCE_SCAN_CHUNK as _default_chunk

    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F)

    baseline = _Leaky(init_hidden=False)
    seq_ref, final_ref = baseline.forward_sequence(x_seq)

    try:
        set_sequence_scan_chunk(1)
        m = _Leaky(init_hidden=False)
        seq1, final1 = m.forward_sequence(x_seq)

        set_sequence_scan_chunk(3)
        m = _Leaky(init_hidden=False)
        seq3, final3 = m.forward_sequence(x_seq)

        assert torch.equal(seq1, seq_ref)
        assert torch.equal(seq3, seq_ref)
        assert torch.allclose(final1, final_ref, atol=1e-6)
        assert torch.allclose(final3, final_ref, atol=1e-6)
    finally:
        set_sequence_scan_chunk(_default_chunk)


def test_set_sequence_scan_chunk_rejects_invalid():
    with pytest.raises(ValueError, match="positive int"):
        set_sequence_scan_chunk(0)
    with pytest.raises(ValueError, match="positive int"):
        set_sequence_scan_chunk(True)


def test_forward_sequence_does_not_mutate_input_state():
    m = _Leaky(init_hidden=False)
    state = m.initial_state((B, F))
    snapshot = [s.clone() for s in state]
    m.forward_sequence(torch.randn(T, B, F), state)
    for orig, snap in zip(state, snapshot, strict=True):
        assert torch.equal(orig, snap)


def test_forward_sequence_hidden_multi_matches_manual():
    torch.manual_seed(0)
    m = _TwoOut(init_hidden=True)
    x_seq = torch.randn(T, B, F)
    seq_o1, seq_o2 = m.forward_sequence(x_seq)

    ref = _TwoOut(init_hidden=True)
    outs1, outs2 = [], []
    for t in range(T):
        o1, o2 = ref(x_seq[t])
        outs1.append(o1)
        outs2.append(o2)
    assert torch.allclose(seq_o1, torch.stack(outs1), atol=1e-6)
    assert torch.allclose(seq_o2, torch.stack(outs2), atol=1e-6)


# ----------------------------------------------------------------------
# H. compile_sequence_scan / fast_sequence_
# ----------------------------------------------------------------------


def test_compile_sequence_scan_hidden_matches_eager():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F)

    compiled = _Leaky(init_hidden=True).compile_sequence_scan(mode="default")
    eager = _Leaky(init_hidden=True)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert torch.allclose(out_c, out_e, atol=1e-5)


def test_compile_sequence_scan_explicit_matches_eager():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F)

    compiled = _Leaky(init_hidden=False).compile_sequence_scan(mode="default")
    eager = _Leaky(init_hidden=False)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert isinstance(out_c, tuple) and isinstance(out_e, tuple)
    assert torch.allclose(out_c[0], out_e[0], atol=1e-5)
    assert torch.allclose(out_c[1], out_e[1], atol=1e-5)


def test_compile_sequence_scan_explicit_state_none_allocates_per_call():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F)

    compiled = _Leaky(init_hidden=False).compile_sequence_scan(mode="default")
    eager = _Leaky(init_hidden=False)

    out1 = compiled.forward_sequence(x_seq)
    out2 = compiled.forward_sequence(x_seq)
    ref = eager.forward_sequence(x_seq)

    # state=None allocates a fresh initial state inside the compiled call, so
    # repeated calls must be mutually independent and match eager.
    assert isinstance(out1, tuple) and isinstance(out2, tuple)
    for a, b in zip(out1, out2, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    for a, b in zip(out1, ref, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_compile_sequence_scan_explicit_provided_state_not_mutated():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F)

    compiled = _Leaky(init_hidden=False).compile_sequence_scan(mode="default")
    eager = _Leaky(init_hidden=False)

    state = eager.initial_state((B, F), device=x_seq.device, dtype=x_seq.dtype)
    state_before = tuple(s.clone() for s in state)

    out = compiled.forward_sequence(x_seq, state)
    ref = eager.forward_sequence(x_seq, state)

    # The scan reads the provided state and returns a new final state; the
    # caller's tensors are never written in place.
    assert isinstance(out, tuple)
    for a, b in zip(out, ref, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    for given, before in zip(state, state_before, strict=True):
        assert torch.equal(given, before)


def test_compile_sequence_scan_multi_matches_eager():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F)

    compiled = _TwoOut(init_hidden=True).compile_sequence_scan(mode="default")
    eager = _TwoOut(init_hidden=True)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert isinstance(out_c, tuple) and isinstance(out_e, tuple)
    for a, b in zip(out_c, out_e, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_compile_sequence_scan_routes_forward_sequence():
    torch._dynamo.reset()
    m = _Leaky(init_hidden=True)
    m.compile_sequence_scan(mode="default")
    assert m._cr_compiled_sequence is not None

    ref = _Leaky(init_hidden=True)
    x_seq = torch.randn(T, B, F)
    out = m.forward_sequence(x_seq)
    ref_out = ref.forward_sequence(x_seq)
    assert out.shape == ref_out.shape
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_compile_sequence_scan_allocates_hidden_before_compiled_call():
    torch._dynamo.reset()
    m = _Leaky(init_hidden=True)
    m.compile_sequence_scan(mode="default")
    assert m._cr_allocated is False

    m.forward_sequence(torch.randn(T, B, F))
    assert m._cr_allocated is True
    assert "mem" in m._buffers
    assert "out" in m._buffers


@pytest.mark.parametrize("t_len", [1, 2, 7, 8, 9, 16])
def test_compile_sequence_scan_edge_lengths_match_eager(t_len):
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(t_len, B, F)

    compiled = _Leaky(init_hidden=False).compile_sequence_scan(mode="default")
    eager = _Leaky(init_hidden=False)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert isinstance(out_c, tuple)
    for a, b in zip(out_c, out_e, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("t_len", [1, 2, 7, 8, 9, 16])
def test_compile_sequence_scan_edge_lengths_hidden_match_eager(t_len):
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(t_len, B, F)

    compiled = _Leaky(init_hidden=True).compile_sequence_scan(mode="default")
    eager = _Leaky(init_hidden=True)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert torch.allclose(out_c, out_e, atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for CUDA-graph (reduce-overhead) compile",
)
def test_compile_reduce_overhead_held_output_survives():
    torch._dynamo.reset()
    m = _Leaky(init_hidden=False).to("cuda")
    m.compile_sequence_scan(mode="reduce-overhead")

    x_seq = torch.randn(3, 2, 8, device="cuda")
    first = m.forward_sequence(x_seq)
    second = m.forward_sequence(x_seq)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for CUDA-graph (reduce-overhead) compile",
)
def test_compile_reduce_overhead_state_none_correct_and_held_survives():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F, device="cuda")

    compiled = (
        _Leaky(init_hidden=False)
        .to("cuda")
        .compile_sequence_scan(mode="reduce-overhead")
    )
    eager = _Leaky(init_hidden=False).to("cuda")

    ref = eager.forward_sequence(x_seq)
    first = compiled.forward_sequence(x_seq)
    second = compiled.forward_sequence(x_seq)

    # state=None allocates the initial state inside the compiled graph; results
    # must match eager, repeated calls must be independent, and graph-mode
    # cloning must keep previously returned tensors (incl. the state) intact.
    assert isinstance(first, tuple)
    for a, b in zip(first, ref, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_compile_clone_gating_default_no_clone(monkeypatch):
    seen = {}

    def fake_compile(fn, **kw):
        def wrapped(x_seq, state):
            r = fn(x_seq, state)
            seen["r"] = r
            return r

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    m = _Leaky(init_hidden=False)
    m.compile_sequence_scan(mode="default")
    out = m.forward_sequence(torch.randn(T, B, F))
    assert out is seen["r"]


def test_compile_clone_gating_reduce_overhead_clones(monkeypatch):
    seen = {}

    def fake_compile(fn, **kw):
        def wrapped(x_seq, state):
            r = fn(x_seq, state)
            seen["r"] = r
            return r

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    m = _TwoOut(init_hidden=False)
    m.compile_sequence_scan(mode="reduce-overhead")
    out = m.forward_sequence(torch.randn(T, B, F))
    assert isinstance(out, tuple)
    assert out is not seen["r"]
    for a, b in zip(out, seen["r"], strict=True):
        assert a is not b
        assert torch.equal(a, b)


def test_fast_sequence_disables_validation():
    m = _Leaky()
    m.fast_sequence_()
    assert m.validate is False


def test_fast_sequence_compiles_scan():
    torch._dynamo.reset()
    m = _Leaky(init_hidden=True)
    m.fast_sequence_()
    assert m._cr_compiled_sequence is not None

    ref = _Leaky(init_hidden=True)
    x_seq = torch.randn(T, B, F)
    out = m.forward_sequence(x_seq)
    ref_out = ref.forward_sequence(x_seq)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_fast_sequence_compile_scan_false():
    m = _Leaky()
    m.fast_sequence_(compile_scan=False)
    assert m.validate is False
    assert m._cr_compiled_sequence is None


def test_fast_sequence_default_mode_is_default(monkeypatch):
    captured = {}

    def fake_compile(fn, **kw):
        captured.update(kw)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    m = _Leaky()
    m.fast_sequence_()
    assert captured["mode"] == "default"

    m2 = _Leaky()
    m2.fast_sequence_(mode="reduce-overhead")
    assert captured["mode"] == "reduce-overhead"


def test_compile_sequence_scan_returns_self(monkeypatch):
    def fake_compile(fn, **kw):
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    m = _Leaky()
    assert m.compile_sequence_scan() is m


# ----------------------------------------------------------------------
# I. get_extra_state / set_extra_state (serialization)
# ----------------------------------------------------------------------


def test_extra_state_none_when_explicit():
    m = _Leaky(init_hidden=False)
    assert m.get_extra_state() is None


def test_extra_state_empty_when_never_allocated():
    m = _Leaky(init_hidden=True)
    assert m.get_extra_state() is None

    snapshot = m.state_dict()
    assert snapshot["_extra_state"] is None


def test_extra_state_contains_hidden_buffers_detached():
    m = _TwoOut(init_hidden=True)
    m(torch.randn(B, F))
    extra = m.get_extra_state()
    assert set(extra.keys()) == {"o1", "o2", "mem"}
    for t in extra.values():
        assert t.requires_grad is False
        assert t.grad_fn is None


def test_state_dict_roundtrip_restores_hidden():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))
    snapshot = m.state_dict()
    m.reset()
    m.load_state_dict(snapshot)
    assert torch.allclose(m._buffers["mem"], snapshot["_extra_state"]["mem"])


def test_set_extra_state_registers_missing_buffer():
    m = _Leaky(init_hidden=True)
    buf = torch.zeros(B, F)
    m.set_extra_state({"mem": buf})
    assert "mem" in m._buffers
    assert m._cr_allocated is True
    assert m._buffers["mem"] is buf


def test_set_extra_state_none_noop():
    m = _Leaky(init_hidden=True)
    m.set_extra_state(None)
    assert m._cr_allocated is False

    m2 = _Leaky(init_hidden=False)
    m2.set_extra_state({"mem": torch.zeros(B, F)})
    assert m2._cr_allocated is False


# ----------------------------------------------------------------------
# J. reset / detach
# ----------------------------------------------------------------------


def test_reset_noop_when_explicit():
    m = _Leaky(init_hidden=False)
    m.reset()
    assert m._cr_allocated is False


def test_reset_restores_defaults_hidden():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))
    m.reset()
    assert torch.allclose(m._buffers["mem"], torch.zeros(B, F))
    assert torch.allclose(m._buffers["out"], torch.zeros(B, F))


def test_reset_uses_spec_default_not_zero():
    m = _CallableDefault(init_hidden=True)
    m(torch.randn(B, F))
    m.reset()
    assert torch.allclose(m._buffers["mem"], torch.full((B, F), 0.25))


def test_detach_noop_when_explicit():
    m = _Leaky(init_hidden=False)
    m.detach()
    assert m._cr_allocated is False


def test_detach_breaks_graph_hidden():
    m = _Leaky(init_hidden=True)
    x = torch.randn(B, F, requires_grad=True)
    m(x)
    assert m._buffers["mem"].requires_grad is True
    m.detach()
    assert m._buffers["mem"].requires_grad is False
    assert m._buffers["mem"].grad_fn is None


# ----------------------------------------------------------------------
# K. __repr__ / __signature__
# ----------------------------------------------------------------------


def test_repr_includes_size_and_init_hidden():
    r = repr(_Leaky(size=16, init_hidden=True))
    assert "size=16" in r
    assert "init_hidden=True" in r


def test_size_is_metadata_only_does_not_affect_allocation():
    m = _Leaky(size=16, init_hidden=True)
    x = torch.randn(3, 7)
    m(x)
    assert m._buffers["mem"].shape == x.shape


def test_repr_omits_size_when_none():
    r = repr(_Leaky())
    assert "size=" not in r
    assert "init_hidden=False" in r


def test_signature_standard_params():
    sig = inspect.signature(_Leaky)
    names = list(sig.parameters)
    assert names[0] == "self"
    assert sig.parameters["size"].default is None
    assert sig.parameters["init_hidden"].default is False
    assert sig.parameters["validate"].default is None
    assert sig.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD


def test_signature_param_kwargs():
    sig = inspect.signature(_Leaky)
    assert sig.parameters["beta"].default == 0.5
    assert sig.parameters["learnable_beta"].default is False
    assert sig.parameters["force_learn_beta"].default is False
    assert sig.parameters["beta_constraint"].default is clamp_unit_interval


def test_signature_extra_init_params_only_for_snn():
    class _Plain(CrModule):
        class Specs:
            o = CrModule.OutputSpec()

        def _step(self, x):
            return x

    class _Snn(SnnModule):
        class Specs:
            o = SnnModule.OutputSpec()
            s = SnnModule.StateSpec()

        def _step(self, x, s):
            return x, x

    plain_sig = inspect.signature(_Plain)
    assert "reset_mechanism" not in plain_sig.parameters
    assert "spike_grad" not in plain_sig.parameters

    snn_sig = inspect.signature(_Snn)
    assert "reset_mechanism" not in snn_sig.parameters
    assert "spike_grad" in snn_sig.parameters


# ----------------------------------------------------------------------
# N. Constant `overridable`
# ----------------------------------------------------------------------


class _OpenConst(CrModule):
    k: int = CrModule.Constant(3, dtype=int)

    class Specs:
        o = CrModule.OutputSpec()

    def _step(self, x):
        return (x,)


class _LockedConst(CrModule):
    rank: int = CrModule.Constant(2, dtype=int, overridable=False)

    class Specs:
        o = CrModule.OutputSpec()

    def _step(self, x):
        return (x,)


class _LockedChild(_LockedConst):
    pass


class _LockedRedeclare(_LockedConst):
    rank: int = CrModule.Constant(4, dtype=int)


def test_constant_overridable_defaults_true():
    assert _OpenConst._cr_constant_specs["k"].overridable is True
    assert _OpenConst(k=7).k == 7
    assert "k" in inspect.signature(_OpenConst).parameters


def test_constant_locked_rejects_kwarg():
    m = _LockedConst()
    assert m.rank == 2
    assert _LockedConst._cr_constant_specs["rank"].overridable is False

    with pytest.raises(TypeError, match="not overridable"):
        _LockedConst(rank=4)


def test_constant_locked_hidden_from_signature():
    assert "rank" not in inspect.signature(_LockedConst).parameters


def test_constant_lock_inherited_when_unset():
    assert _LockedChild._cr_constant_specs["rank"].overridable is False
    assert _LockedChild().rank == 2

    with pytest.raises(TypeError, match="not overridable"):
        _LockedChild(rank=4)


def test_constant_locked_subclass_may_redeclare_default():
    assert _LockedRedeclare._cr_constant_specs["rank"].overridable is False
    assert _LockedRedeclare().rank == 4

    with pytest.raises(TypeError, match="not overridable"):
        _LockedRedeclare(rank=2)


def test_constant_reopen_locked_raises_at_class_creation():
    with pytest.raises(TypeError, match="reopens a locked"):

        class _Reopen(_LockedConst):
            rank: int = CrModule.Constant(2, dtype=int, overridable=True)


# ----------------------------------------------------------------------
# L. Behaviors ported from orig/ that still apply
# ----------------------------------------------------------------------


def test_hard_zero_reset_equals_zero_reset_on_binary():
    mem = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]])
    spk = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    assert torch.equal(
        mem.masked_fill(spk > 0, 0.0),
        mem * (1.0 - spk),
    )


def test_hidden_buffer_store_keeps_registration():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))
    m(torch.randn(B, F))
    assert "mem" in m._buffers


def test_multi_output_step_state_matches_forward():
    m = _TwoOut(init_hidden=False)
    state = m.initial_state((B, F))
    x = torch.randn(B, F)

    outputs, next_state = m.step_state(x, state)
    fwd = m.forward(x, *state)
    assert isinstance(fwd, tuple)
    assert torch.allclose(outputs[0], fwd[0])
    assert torch.allclose(outputs[1], fwd[1])
    assert torch.allclose(next_state[0], fwd[2])


def test_sequence_consistency_compiled_vs_loop():
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = _Leaky(init_hidden=False)
    m.compile_sequence_scan(mode="default")

    x_seq = torch.randn(T, B, F)
    seq, final = m.forward_sequence(x_seq)

    ref = _Leaky(init_hidden=False)
    state = ref.initial_state((B, F))
    outs = []
    for t in range(T):
        spk, state = ref.step_state(x_seq[t], state)
        outs.append(spk)
    assert torch.allclose(seq, torch.stack(outs), atol=1e-5)
    assert torch.allclose(final, state[0], atol=1e-5)


def test_state_dict_does_not_include_hidden_buffers_as_plain_keys():
    m = _Leaky(init_hidden=True)
    m(torch.randn(B, F))
    sd = m.state_dict()
    assert "mem" not in sd
    assert "out" not in sd
    assert "_extra_state" in sd


# ----------------------------------------------------------------------
# M. Generic recurrence (non-SNN) on the base class
# ----------------------------------------------------------------------


def test_generic_rnn_needs_no_spike_grad_or_reset_machinery():
    m = _PlainRNN()
    assert not hasattr(m, "spike_grad")
    assert not hasattr(m, "_cr_apply_resets")
    assert not hasattr(m, "_cr_reset_exprs")


def test_generic_rnn_post_step_identity_keeps_first_output():
    m = _PlainRNN(init_hidden=False)
    x = torch.randn(B, F)
    state = m.initial_state((B, F))
    out = m(x, *state)
    assert isinstance(out, tuple) and len(out) == 2
    expected = torch.tanh(0.5 * state[0] + x)
    assert torch.allclose(out[0], expected)
    assert torch.allclose(out[1], expected)


def test_generic_rnn_works_hidden_and_explicit():
    torch.manual_seed(0)
    hidden = _PlainRNN(init_hidden=True)
    explicit = _PlainRNN(init_hidden=False)
    state = explicit.initial_state((B, F))

    for _ in range(T):
        x = torch.randn(B, F)
        h_out = hidden(x)
        e_out = explicit(x, *state)
        assert isinstance(h_out, torch.Tensor)
        assert torch.allclose(h_out, e_out[0], atol=1e-6)
        state = e_out[1:]
        assert torch.allclose(hidden._buffers["h"], state[0], atol=1e-6)


def test_generic_rnn_forward_sequence_works():
    torch.manual_seed(0)
    m = _PlainRNN(init_hidden=True)
    out = m.forward_sequence(torch.randn(T, B, F))
    assert out.shape == (T, B, F)


# ----------------------------------------------------------------------
# I. Review fixes: graph-mode guard, state-only alloc, constrained_dict
# ----------------------------------------------------------------------


def test_compile_sequence_scan_refuses_graph_modes_in_hidden_mode():
    # Hidden-mode compiled scans store outputs into module buffers; under
    # CUDA-graph modes those are graph-owned and alias recycled memory.
    # The setup must refuse loudly instead of hedging in docstrings.
    m = _Leaky(init_hidden=True)

    with pytest.raises(ValueError, match="reduce-overhead"):
        m.compile_sequence_scan(mode="reduce-overhead")

    with pytest.raises(ValueError, match="max-autotune"):
        m.compile_sequence_scan(mode="max-autotune")

    with pytest.raises(ValueError, match="fast_sequence_"):
        m.fast_sequence_(mode="reduce-overhead")

    # Explicit mode stays eligible for graph modes (outputs are cloned on
    # return); setting up the wrapper must not raise.
    e = _Leaky(init_hidden=False)
    e.compile_sequence_scan(mode="reduce-overhead")
    assert e._cr_compiled_sequence is not None


def test_alloc_hidden_creates_state_buffers_only():
    # Output buffers used to be pre-allocated in the input's shape and then
    # immediately replaced by the first store: dead work with a lying shape.
    class _PooledOut(CrModule):
        class Specs:
            out = CrModule.OutputSpec()
            mem = CrModule.StateSpec()

        def _step(self, x, mem):
            mem = 0.5 * mem + x
            return mem.mean(dim=-1, keepdim=True), mem

    m = _PooledOut(init_hidden=True)
    m.allocate_like(torch.randn(3, 7))

    assert m._buffers["mem"].shape == (3, 7)
    assert m._buffers.get("out") is None

    y = m(torch.randn(3, 7))

    assert tuple(y.shape) == (3, 1)
    assert tuple(m._buffers["out"].shape) == (3, 1)

    # Output buffers remain non-persistent after first store.
    sd = m.state_dict()
    assert "out" not in sd and "mem" not in sd


def test_constrained_dict_matches_constrain_by_name():
    net = LIF(learnable_beta=True, learnable_threshold=True)

    named = net.constrained_dict()
    assert set(named) == {"beta", "threshold"}

    assert torch.equal(named["beta"], net.constrain("beta"))
    assert torch.equal(named["threshold"], net.constrain("threshold"))

    # Constraints are applied by name too (beta clamps to [0, 1]).
    assert (named["beta"] >= 0).all() and (named["beta"] <= 1).all()


def test_deepcopy_drops_compiled_scan():
    # _cr_compiled_sequence closes over the original module; without this
    # the copy would silently run the original's dynamics.
    m = LIF(beta=0.9)
    m.fast_sequence_()
    assert m._cr_compiled_sequence is not None

    c = copy.deepcopy(m)
    assert c._cr_compiled_sequence is None

    c.beta.data.fill_(0.1)
    x = torch.randn(5, 2, 4)
    got = c.forward_sequence(x, c.initial_state_for_sequence(x))

    ref = LIF(beta=0.1)
    exp = ref.forward_sequence(x, ref.initial_state_for_sequence(x))
    assert torch.equal(got[0], exp[0])
    assert torch.equal(got[1], exp[1])


def test_copy_drops_compiled_scan_but_shares_params():
    m = LIF(beta=0.9)
    m.fast_sequence_()

    c = copy.copy(m)
    assert c._cr_compiled_sequence is None
    assert c.beta is m.beta
