# Quantum Query Decision

Code for the paper *Fidelity-Aware Execution of Quantum Search Queries: Choosing Between
Circuit Cutting and Predicate Factorization on NISQ Hardware*.

A simulation-free, closed-form criterion that reads compiled gate counts and device
calibration to choose how to run a quantum search query: whole, cut, or factorized.
Validated on IBM Heron hardware (ibm_fez, ibm_marrakesh, ibm_kingston).

## Structure

```
qqc/fidelity_model.py   Closed-form log10-fidelity from gate counts + routing penalty
qqc/cut_decision.py     Intact-vs-cut comparison on the shots-to-precision scale
qqc/optimizer.py        Width-sweep cut-plan cost
qqc/factorize.py        Separability test, factorized cost, strategy choice
qqc/predicates.py       SQL predicates -> matching sets -> amplitude-amplification oracles
qqc/islands.py          Low-noise island detection from calibration
experiments/            Runnable experiments (see table below) and FakeFez calibration
experiments/output/     Generated tables, JSON results, and hardware-run outputs
configs/                IBM credentials (hardware.py, gitignored; see hardware.py.example)
tests/                  Pure-Python tests for the planner internals
demo.py                 Whole/cut/factorize decision example (no quantum deps)
```

## Scripts

| Script | Paper | Description |
|--------|-------|-------------|
| `run_planning.py` | §5.1, 5.6 | Routing-penalty, selectivity, and scaling tables (closed form) |
| `run_hw_validation.py` | §5.2 | Predicted vs measured fidelity over a query suite |
| `run_factorize.py` | §5.3 | Whole vs factorized retrieval for one conjunctive query |
| `run_factorize_matrix.py` | §5.3 | Factorization across widths/groups, batched into one job |
| `run_real_cut.py` | §5.4 | Real qiskit-addon-cutting reconstruction (n=2 anchor) |
| `run_crossdevice.py` | §5.5 | Factorization across several physical devices |
| `run_robustness.py` | §5.5 | Routing penalty across calibrations and circuit classes |
| `run_tpch.py` | §5.6 | TPC-H-style predicates verified against classical SQL |
| `run_localdiff.py` | §6 | Local diffusion does not amplify a conjunctive target |
| `make_hw_figure.py` | Fig. | Predicted-vs-measured figure |

## Quick start

```bash
pip install -r requirements.txt

python demo.py            # whole/cut/factorize decision, no quantum deps
bash reproduce.sh         # all simulation results (qiskit + aer)
```

## Hardware

Copy `configs/hardware.py.example` to `configs/hardware.py` and add an IBM Quantum
token and instance (or set `IBM_QUANTUM_TOKEN` / `IBM_CLOUD_INSTANCE`). Then pass
`--hardware`:

```bash
python experiments/run_factorize_matrix.py --hardware default
python experiments/run_tpch.py --hardware default
python experiments/run_crossdevice.py --devices ibm_fez ibm_marrakesh ibm_kingston
```

Without `--hardware`, experiments run on the FakeFez noise model.

## Requirements

Python >= 3.10. See `requirements.txt`.
