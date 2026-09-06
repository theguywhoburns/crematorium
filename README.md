# crematorium

[![CI](https://github.com/theguywhoburns/crematorium/actions/workflows/test.yml/badge.svg)](https://github.com/theguywhoburns/crematorium/actions/workflows/test.yml)

A declarative spiking neural network library for PyTorch research. Neurons
are described by plain nested classes (`Params`, `Specs`) and a single
`_step`; the rest - parameter creation, constraints, resets, hidden state,
and sequence scans - is automatic.

Plain PyTorch only. No custom CUDA kernels; the speed comes from
`torch.compile` over pure Python math. Sequence scans also hold ~30% less
memory than the snnTorch/Norse equivalents on identical hardware (see
[Benchmarks](#benchmarks)).

## Install

`crematorium` uses `torch>=2.13.0`, selected via an extra matching your
backend(example, currently the library is **NOT RELEASED** on PiPy):

```bash
pip install crematorium[cu130]   # NVIDIA / CUDA 13.0
pip install crematorium[cpu]     # CPU only
pip install crematorium[rocm]    # AMD / ROCm 7.2
```

The extras are mutually exclusive: pick exactly one. The `rocm` extra also
pulls the ROCm triton builds (`pytorch-triton-rocm`, `triton-rocm`) needed for
compiled scans on AMD.

> **Note:** the `[tool.uv.sources]` index selection for `torch` is `uv`-only;
> plain `pip` will resolve `torch` from the default PyPI index regardless of
> the extra. Use `uv` for the backend-pinned installs above.

## How it works

Subclass `SnnModule`, declare `Params` and `Specs`, implement `_step`:

```python
from crematorium import clamp_positive, clamp_unit_interval
from crematorium.snn import Reset, SnnModule


class MyLIF(SnnModule):
    class Params:
        beta = SnnModule.Param(0.9, constraint=clamp_unit_interval)
        threshold = SnnModule.Param(1.0, constraint=clamp_positive)

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.subtract("threshold"))

    def _step(self, x, mem):
        beta = self.constrain("beta")
        threshold = self.constrain("threshold")
        mem = beta * mem + x
        spk = self.spike_grad(mem - threshold)
        return spk, mem
```

From that declaration the framework derives everything else:

- `Params` entries become constructor kwargs and `nn.Parameter`s. Each has
  `learnable_<name>`, `force_learn_<name>`, and `<name>_constraint`
  overrides. Read values in `_step` with `self.constrain(name)` (one value
  by name) or `self.constrained_dict()` (the whole mapping). Constraints
  apply **only to learnable params**: a fixed param is used raw, a learnable
  one is clamped every step.
- `Specs` entries are `OutputSpec` (returned, not recurrent) or `StateSpec`
  (recurrent state, threaded through `_step`). `StateSpec(shape=...)` can
  decouple a state shape from the input shape, and can follow any named
  input of a multi-input module.
- `_step(x, *state)` returns `(spike, *states)`; the first element is the
  spike output and declarative resets (`Reset.subtract` / `zero` /
  `hard_zero` / `set` / `add` / `custom`) are applied to the states
  afterwards, in both execution modes.
- `Constant` declares a non-learnable hyperparameter (e.g. `dt`), with an
  optional `validate=` callable.
- `forward`, `step_state`, the state factories (`initial_state`,
  `zero_state`, `allocate_like`, ...), and `forward_sequence` are derived;
  you don't write them.
- Multi-input modules declare a nested `Inputs` class and `_step` receives
  the inputs positionally (e.g. `MCN`'s basal and apical inputs — see
  `src/crematorium/snn/neurons/MCN.py`). Call sites accept a single tensor, a
  tuple/list, or a dict keyed by input name.

## Running it

Time-major sequences `(time, batch, features)`:

```python
import torch
from crematorium.snn import LIF

x = torch.randn(1000, 32, 64)

lif = LIF()
state = lif.initial_state_for_sequence(x)
spikes, mem = lif.forward_sequence(x, state)
```

Per-step with explicit state:

```python
spk, (mem,) = lif.step_state(x[0], lif.zero_state((32, 64)))
```

Hidden mode (the module owns its state buffers):

```python
lif = LIF(init_hidden=True)
spikes = lif.forward_sequence(x)  # state lives in module buffers
```

Compiled scan (compiles the whole sequence into one graph):

```python
lif.fast_sequence_()
spikes = lif.forward_sequence(x)
```

Multilayer stacks compose `nn.Module` layers and neurons; the whole stack
scans and compiles as one unit (see [docs/sequential.md](docs/sequential.md)):

```python
from crematorium.nn import Sequential
import torch.nn as nn

net = Sequential(nn.Linear(64, 64), LIF(), nn.Linear(64, 10), LIF())
```

Note: `Sequential` threads a single tensor, so multi-input modules (like
`MCN`) cannot be stacked inside it — compose those manually or with a
branching container.

Training (explicit mode, backprop through time over `forward_sequence`):

```python
lif = LIF(init_hidden=False, learnable_beta=True, learnable_threshold=True)
opt = torch.optim.Adam(lif.parameters(), lr=1e-2)

for epoch in range(100):
    opt.zero_grad()

    x = torch.randn(T, B, F)
    spikes, _ = lif.forward_sequence(x, lif.zero_state((B, F)))
    rate = spikes.mean(dim=(0, 2))  # mean firing rate per sample

    target = torch.where(x.mean(dim=(0, 2)) > 0, 1.0, 0.0)
    loss = torch.nn.functional.mse_loss(rate, target)
    loss.backward()
    opt.step()
```

See [examples/sequence_training.ipynb](examples/sequence_training.ipynb) for a
complete worked example.

## Execution modes

- **Explicit** (`init_hidden=False`, default): the caller owns state. Each
  call is a pure function of `(x, *state)`. Use this for training.
- **Hidden** (`init_hidden=True`): the module owns state in internal
  buffers. Use this for inference and long-running generation.

Hidden-mode training needs one extra call: buffers retain the autograd
graph after `forward_sequence`, so for chunked BPTT call `net.detach()`
between chunks (carries state values forward, cuts the graph, ~1µs).
Without it the second `backward()` fails. Details in
[docs/execution-modes.md](docs/execution-modes.md).

## Configuration

- **Surrogate gradient** - `spike_grad=` at construction swaps the default
  straight-through estimator. Available: `sigmoid_surrogate(beta)`,
  `atan_surrogate(beta)`, `triangular_surrogate(beta)`,
  `fast_sigmoid_surrogate(beta)`, `straight_through_surrogate`. Params whose
  drift would invalidate a frozen surrogate are marked
  `frozen_surrogate=True` (e.g. `MCN`'s `tau_L`); with a marked param
  learnable, the constructor refuses an explicit `spike_grad` rather than
  train on silently frozen math — keep the param fixed, or pass a callable
  that reads the live value.
- **Validation** - `validate=True` (default) checks step arity and state
  shapes on each call; override per module, or globally with
  `set_validation()` / `no_validation()`. Cost is ~10-20% in eager mode
  and ~zero compiled. Disable it only for long-running, stable training
  runs, where you are sure NOTHING will change and everything will be stable.
- **Long sequences** - the compiled scan is a fully unrolled T-step graph,
  so compilation cost and memory grow with T (fast up to ~T=1000,
  impractical past ~T=3000). Chunk the input for very long sequences, or
  tune the eager scan with `set_sequence_scan_chunk(n)`.
- **CUDA graphs** - supported in explicit mode (`init_hidden=False`;
  outputs are cloned on return under `mode="reduce-overhead"` /
  `"max-autotune"`). Hidden mode refuses graph modes with a clear error:
  its buffers must not alias compiler-recycled graph memory. In hidden
  mode still call `allocate_like(x)` once before the first compiled call,
  so buffer allocation stays outside the traced region.

## Included neurons and utilities

- Neurons: `LIF`, `AdEx`, `ALIF`, `HH`, `Izhikevich`, `SRM0`,
  `TwoCompartment`, `MCN` (three-compartment basal/apical/soma).
- `SpikeTrain` (`crematorium.util.SpikeTrain`): quantile fractions to
  Poisson population spike trains via Gaussian receptive fields, plus
  Poisson and latency encoders and `.custom(...)`; dense or event-packed
  GPU form. See [examples/spike_train.py](examples/spike_train.py).

## Docs

- [Execution modes: hidden vs explicit](docs/execution-modes.md)
- [Resets](docs/resets.md)
- [Constraints](docs/constraints.md)
- [Sequence scans](docs/sequence-scan.md)
- [Sequential networks](docs/sequential.md)
- [Publishing to PyPI](docs/publishing.md)

## Benchmarks

Measured on `NVIDIA GeForce RTX 3050 Laptop GPU 4G` (single consumer
laptop GPU, best-of steady-state numbers — one data point, not evidence).
Crematorium rows use `validate=False` (snnTorch/Norse have no equivalent
toggle). Tables below are LIF, T=1000, B=32, F=1024:

- Sequence mode uses ~30% less memory than the snnTorch/Norse equivalents
  here (254 vs ~378 MiB) — relevant on 4-8GB laptop GPUs.

| library         | mode          | compiled | ms      | steps/s | peak MiB | vs crematorium seq eager |
| --------------- | ------------- | -------- | ------- | ------- | -------- | ------------------------ |
| crematorium LIF | seq hidden    | eager    | 39.918  | 25,051  | 254.6    | 1.00x                    |
| crematorium LIF | seq hidden    | compile  | 3.451   | 289,810 | 252.5    | 0.09x                    |
| crematorium LIF | seq explicit  | eager    | 41.180  | 24,284  | 254.8    | 1.03x                    |
| crematorium LIF | seq explicit  | compile  | 3.420   | 292,401 | 252.4    | 0.09x                    |
| crematorium LIF | step hidden   | eager    | 45.254  | 22,098  | 126.8    | 1.13x                    |
| crematorium LIF | step hidden   | compile  | 38.433  | 26,020  | 126.5    | 0.96x                    |
| crematorium LIF | step explicit | eager    | 41.608  | 24,034  | 127.0    | 1.04x                    |
| crematorium LIF | step explicit | compile  | 36.418  | 27,459  | 126.8    | 0.91x                    |
| snntorch        | seq           | eager    | 101.685 | 9,834   | 377.5    | 2.55x                    |
| snntorch        | seq           | compile  | 39.432  | 25,360  | 378.0    | 0.99x                    |
| norse           | seq           | eager    | 100.005 | 9,999   | 378.3    | 2.51x                    |
| norse           | seq           | compile  | 4.332   | 230,829 | 376.6    | 0.11x                    |
| norse           | step          | eager    | 99.913  | 10,009  | 128.0    | 2.50x                    |
| norse           | step          | compile  | 37.215  | 26,871  | 127.3    | 0.93x                    |

Notes on reading this: the compiled scan fuses the unrolled T-step graph
into one call, which is where the speedup over the per-step loop comes
from (per-step Python dispatch dominates, so compiling the loop alone
barely moves it: 38.4 vs 45.3 ms). Sequence mode holds the full
`(T, B, F)` spike stack (~254 MiB here vs ~378 MiB); step-mode memory is
comparable across all three (~127 MiB). Compiled timings vary run-to-run
on laptop GPUs (typical single runs read ~3.3-3.8 ms here).

> **Ratio convention**: `bench_all_vs.py` prints `framework_time /
crematorium_seq_eager_time` as the trailing `(N.Nx)` factor. Below 1.0
> means faster than the first measured row (`seq hidden eager`).

Multilayer network (`Linear(512,512) -> LIF x4 -> Linear(512,10) -> LIF`,
T=1000, B=32, F=512):

| library                | mode      | compiled | ms      | steps/s | peak MiB | vs crematorium seq eager |
| ---------------------- | --------- | -------- | ------- | ------- | -------- | ------------------------ |
| crematorium Sequential | seq       | eager    | 217.549 | 4,597   | 74.4     | 1.00x                    |
| crematorium Sequential | seq       | compile  | 22.759  | 43,938  | 79.5     | 0.10x                    |
| crematorium Sequential | step      | eager    | 229.107 | 4,365   | 73.7     | 1.05x                    |
| snntorch               | step loop | eager    | 575.159 | 1,739   | 76.2     | 2.64x                    |
| snntorch               | step loop | compile  | 99.565  | 10,044  | 77.2     | 0.46x                    |
| norse                  | seq       | eager    | 514.664 | 1,943   | 263.2    | 2.37x                    |
| norse                  | seq       | compile  | 25.099  | 39,842  | 260.1    | 0.12x                    |

snnTorch has no sequence module, so it runs the canonical per-step loop
with explicit membrane state (the `step loop` rows); `torch.compile`
stalls on that Python loop, so the eager loop is the comparable row.

Reproduce either table:

```bash
uv run --group bench python benchmarks/bench_all_vs.py --steps 1000
uv run --group bench python benchmarks/bench_multilayer_sequential_vs_all.py --steps 1000
```

Results print to the console and land as CSVs under `benchmarks/results/`.

## Development

```bash
uv run pytest            # run the test suite
uv run pyright src       # type-check (src/ only; tests/benchmarks excluded)
uv run ruff check        # lint (rules: E, F, B, RUF; see pyproject.toml)
```

- Tests: pytest. Type-checking: pyright covers `src/` only (the declarative
  `Param()` is `Any`-typed by design, so the tests/benchmarks surface
  produces ~300 noise diagnostics; see the `exclude` in `[tool.pyright]`).
- Formatting: `ruff format` (run `uv run ruff format .` before committing).
