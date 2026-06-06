#!/usr/bin/env bash
# Reproduce every simulation result in the paper. Hardware runs are listed at the end and
# need an IBM Quantum account in configs/hardware.py; they are not run here.
#
# Usage:  bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== environment =="
python3 -c "import qiskit,qiskit_aer,qiskit_addon_cutting,scipy,networkx; \
print('qiskit',qiskit.__version__,'| aer',qiskit_aer.__version__,\
'| cutting',qiskit_addon_cutting.__version__)"

echo "== unit tests =="
python3 -m pytest tests/ -q

echo "== analytical planner tables (routing, selectivity, scaling) =="
python3 experiments/run_planning.py

echo "== robustness: cross-device + across circuit classes =="
python3 experiments/run_robustness.py

echo "== predicate factorization (recovers a routing-destroyed query) =="
python3 experiments/run_factorize.py --na 3 --nb 3
python3 experiments/run_factorize.py --na 4 --nb 4

echo "== factorization scaling by full noisy simulation (Table 5) =="
python3 experiments/run_scaling_sim.py

echo "== factorization matrix: varied widths/groups/marks, layout spread (sim subset) =="
python3 experiments/run_factorize_matrix.py --max-n 9

echo "== end-to-end TPC-H-style predicates, executed + verified vs classical SQL =="
python3 experiments/run_tpch.py

echo "== local diffusion is not a baseline for conjunctive retrieval =="
python3 experiments/run_localdiff.py

echo "== criterion vs hardware (dry run on Aer) + recalibration =="
python3 experiments/run_hw_validation.py --backend FakeFez

echo "== decision cost: criterion vs noisy-sim profiling =="
python3 experiments/run_hw_validation.py --speedup

echo "== real cut + reconstruct (n=2 anchor) =="
python3 experiments/run_real_cut.py

echo
echo "All simulation results are under experiments/output/."
echo "Hardware runs (need configs/hardware.py; run manually):"
echo "  python3 experiments/run_hw_validation.py --hardware default --repeats 3   # criterion vs chip"
echo "  python3 experiments/run_factorize.py --hardware default                   # factorize vs whole on chip"
echo "  python3 experiments/run_factorize_matrix.py --hardware default            # full factorization matrix, one batched job"
echo "  python3 experiments/run_tpch.py --hardware default                        # TPC-H predicates, one batched job"
echo "  python3 experiments/run_crossdevice.py                                    # factorize on 3 real Heron chips"
echo "  python3 experiments/run_real_cut.py --max-n 3 --num-samples 200000        # cutting wall"
