<div align="center">

# FPRL: Graph-Transformer-Aided Reinforcement Learning for Dynamic Heterogeneous Multi-UAV Task Allocation with Temporal Dependencies

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Journal](https://img.shields.io/badge/MDPI-Drones%202026-b31b1b.svg)](https://www.mdpi.com/journal/drones)

**Zumin Wan, Tao Chen\*, Bangbang Ren, Ruobin Li**

National Key Laboratory of Information Systems Engineering<br/>
National University of Defense Technology, Changsha 410005, China

</div>

---

## 📝 Abstract

Coordinating heterogeneous unmanned aerial vehicles (UAVs) is essential for complex missions such as search and rescue, reconnaissance, and persistent surveillance, since no single platform provides the coverage, redundancy, and complementary capabilities these missions demand. The central difficulty is deciding online which UAV should execute which task, when tasks arrive during execution and when one task's feasibility depends on the start or completion of others — coupling dynamic arrivals, typed temporal dependencies, and heterogeneous capabilities in ways existing methods rarely address jointly.

**FPRL** is a graph-transformer-aided reinforcement learning framework that encodes the mission state as a relational graph over tasks, UAVs, and their pairwise relations, and learns an allocation policy with proximal policy optimization (PPO). Hard action masking, capacity-aware candidate control, and ordered commitment ensure feasible online execution. Against planning-based, heuristic, online-distributed, and learned baselines, FPRL achieves significantly higher verified task completion and oracle-relative completion on a 229-scene benchmark with solver-verified optimal references, converges more stably across seeds, and retains its margin as scenarios scale from 12 to 40 UAVs, despite using half the learned baseline's interaction budget.

> **Terminology note.** In the source code, file names, environment variables, and dataset paths, the abbreviation `mrta` / `MRTA` refers to exactly the same problem as **MUTA** (multi-UAV task allocation) in the paper. The code has deliberately not been renamed, so that the released artifacts stay byte-for-byte reproducible.

---

## 📢 News

- **[2026-01]**: Code repository `FPRL` released with the full paper-reproduction pipeline. 🎉
- **[2026-01]**: The 229-scene CP-SAT-audited benchmark and the s1–s8 scale suite are bundled in `data/`.
- **[2026-01]**: Raw evaluation artifacts for the main table, ablation study, MAPPO baseline, and scale suite are bundled in `results/`.

---

## 🛠 Installation

### Prerequisites

- Python **3.10+** (a conda environment is strongly recommended)
- PyTorch **2.0+** (the paper used `torch==2.1.2` with CUDA 11.8; a CPU-only build also works)
- A GPU is **optional** — `torch_runtime.py` probes CUDA with a real tensor operation and silently falls back to CPU when the device/driver/architecture combination is incompatible (e.g. newer `sm_120` GPUs on older CUDA builds)

### Option 1: Using Conda (Recommended)

```bash
# 1. create an isolated environment
conda create -n fprl python=3.10 -y
conda activate fprl

# 2. install PyTorch matching your CUDA build (example: CUDA 11.8)
pip install --index-url https://download.pytorch.org/whl/cu118 \
            --extra-index-url https://pypi.org/simple torch==2.1.2

# 3. install the remaining dependencies
pip install -r requirements.txt
```

### Option 2: Using pip

```bash
python -m venv fprl
fprl\Scripts\activate          # Windows
# source fprl/bin/activate     # Linux / macOS

pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch, numpy, pandas, matplotlib, ortools, tensorboard, rliable; \
           print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# the policy + baseline packages must import from the repository root
python -c "from baseline.legacy_adapter import LegacyBaselineAdapter; print('baseline OK')"
```

### Runtime switches

| Variable | Effect |
|---|---|
| `MRTA_DEVICE=cuda` | Force GPU execution |
| `MRTA_DEVICE=cpu` | Force CPU execution |
| `KMP_DUPLICATE_LIB_OK=TRUE` | Silence the benign Anaconda `OMP: Error #15` message on Windows |

> **Windows note:** `OMP: Error #15: Initializing libiomp5md.dll ...` appears at the *end* of a run and is caused by two OpenMP runtimes being linked (Anaconda + PyTorch). It does not affect the results; set `KMP_DUPLICATE_LIB_OK=TRUE` to suppress it.

---

## 📂 Data

All benchmarks are bundled in the repository and are **read-only** — you never need to regenerate them to reproduce the paper.

| Path | Content |
|---|---|
| `data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1/` | The 229-scene strict fixed benchmark used for the main comparison and the ablation study. Difficulty is balanced over `[0.20, 0.80]`. Each scene is a folder `env_0` … `env_228` holding a `config.json`. |
| `…/oracle_episode_metrics.csv` | Per-scene CP-SAT reference values used to compute `cpsat_relative_done`. |
| `…/overview.csv` | Human-readable scene summary (UAVs, tasks, edges, difficulty bin). |
| `…/selection_manifest.json` | Record of how the 229 scenes were selected. |
| `data/scale_suite_fixed_sets/s1 … s8/` | Fixed sets for the scale-robustness study: 12 → 40 UAVs and 24 → 80 tasks, 35 scenes per scale. |

### Dataset structure

```
data/
├── MRTAFixedTestSet_oracle229_d020_080_balanced_v1/
│   ├── env_0/config.json
│   ├── env_1/config.json
│   ├── ...
│   ├── env_228/config.json
│   ├── oracle_episode_metrics.csv
│   ├── overview.csv
│   └── selection_manifest.json
└── scale_suite_fixed_sets/
    ├── s1/ ... s8/
    │   └── env_*/config.json
    └── ...
```

### Regenerating the benchmark (optional)

```bash
python generate_fixed_mrta_testset.py --help

# recompute / extend the CP-SAT reference values for a fixed set
python evaluate_cpsat_oracle.py \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --output_csv data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1/oracle_episode_metrics.csv
```

---

## 📦 Model Zoo

Trained checkpoints are **not** bundled with this repository (they total ≈220 MB; the folder is kept source + data + results oriented). The table below documents exactly which directories the evaluation commands expect, so you can drop downloaded weights into `checkpoints/` with the matching layout.

| Checkpoint directory | Variant | Seeds | Notes |
|---|---|---|---|
| `checkpoints/ablation_final_bottleneck/full/seed_{1,2,3}.pt` | **Full method (FPRL)** | 1, 2, 3 | Trained for 1×10⁶ environment steps |
| `checkpoints/ablation_final_bottleneck/wo_rgat/seed_{1,2,3}.pt` | Ablation: w/o GTE (graph transformer encoder) | 1, 2, 3 | |
| `checkpoints/ablation_final_bottleneck/self_only_graph/seed_{1,2,3}.pt` | Ablation: self-only graph (parameter-matched control) | 1, 2, 3 | |
| `checkpoints/ablation_final_bottleneck/wo_mask/seed_{1,2,3}.pt` | Ablation: w/o hard action masking | 1, 2, 3 | |
| `checkpoints/ablation_final_bottleneck/wo_commitment/seed_{1,2,3}.pt` | Ablation: w/o ordered commitment | 1, 2, 3 | |
| `checkpoints/mappo_fair_baseline/full/seed_{1,2,3}.pt` | Fair MAPPO baseline (full, non-sparsified features) | 1, 2, 3 | 2×10⁶ steps |
| `checkpoints/scale_suite_ours_mixed_scale/checkpoint_best_fixed_eval_seed{1,2,3}.pt` | FPRL for the scale suite | 1, 2, 3 | |
| `checkpoints/mappo_scale_1m_paperfair/full/seed_{1,2,3}.pt` | MAPPO for the scale table | 1, 2, 3 | 1×10⁶ steps (matched budget) |

> **Note:** a checkpoint records its own `mrta_config`, so `evaluate_mrta.py` can restore the exact environment configuration it was trained with (`--use_checkpoint_env_config`).

---

## 🚀 Quick Start

All commands are run **from the repository root**.

### Evaluate a trained checkpoint

```bash
# full method, seed 1, on the 229-scene strict benchmark
python evaluate_mrta.py \
  --checkpoint checkpoints/ablation_final_bottleneck/full/seed_1.pt \
  --setting iid \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --oracle_metrics_csv data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1/oracle_episode_metrics.csv \
  --policy_decode greedy --policy_explore_eps 0.0 \
  --episodes 229 --seeds 1 \
  --method_alias ours --output_tag eval_full_seed1 \
  --output_dir results/eval_rerun
```

Use `--episodes 2` for a quick smoke test. Artifacts (`episode_metrics.csv`, `aggregate_metrics.csv`, `summary.json`, figures) are written to `results/eval_rerun/eval_full_seed1/`.

### Evaluate the traditional baselines (no checkpoint needed)

```bash
# GA
python evaluate_mrta.py --baseline --baseline_method ga \
  --setting iid \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --episodes 229 --seeds 1 \
  --output_dir results/eval_rerun --output_tag eval_ga

# CBS-TA-PTC (AAMAS-2024 planning-based baseline)
python evaluate_mrta.py --baseline --baseline_method cbs_ta_ptc \
  --baseline_timeout_sec 5.0 --baseline_episode_timeout_sec 120 \
  --setting iid \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --episodes 229 --seeds 1 \
  --output_dir results/eval_rerun --output_tag eval_cbs

# CTAS-D (online-distributed baseline)
python evaluate_mrta.py --baseline --baseline_method ctas_d \
  --setting iid \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --episodes 229 --seeds 1 \
  --output_dir results/eval_rerun --output_tag eval_ctasd
```

`--baseline_method` accepts `ga`, `cbs_ta_ptc`, and `ctas_d`. By default baselines replan at every decision step; add `--baseline_planned_once` to run them from a single initial plan.

### Train from scratch

```bash
python train_mrta.py \
  --total_steps 1000000 --rollout_steps 768 \
  --num_envs 12 --update_epochs 3 --minibatch_size 384 --env_max_steps 800 \
  --env_backend subproc --no_auto_tune_runtime \
  --seed 1 --save_path checkpoints/ours_full_seed1.pt
```

The training protocol reported in the paper: 1×10⁶ environment steps per seed for FPRL, 2×10⁶ for MAPPO, 12 parallel subprocess environments, rollout length 768, 3 update epochs, minibatch 384, and training scenes restricted to the difficulty interval `[0.20, 0.80]`.

---

## 📊 Results

### Main comparison — 229-scene strict fixed benchmark

Method-level means; inference time is the cumulative action-selection time per episode (three seeds).

| Method | Return ↑ | Recognized Completion ↑ | Oracle-Relative Completion ↑ | Violation Rate ↓ | Inference Time (s) ↓ |
|---|---|---|---|---|---|
| **FPRL (Ours)** | **842.26** | **0.497** | **0.556** | 0.217 | 3.55 |
| MAPPO | 783.75 | 0.440 | 0.464 | 0.205 | **2.14** |
| GA | 626.57 | 0.331 | 0.351 | **0.063** | 29.63 |
| CBS-TA-PTC | 633.92 | 0.430 | 0.475 | 0.197 | 19.79 |
| CTAS-D | 236.92 | 0.139 | 0.147 | 0.174 | 27.11 |

FPRL attains the highest return, recognized completion, and oracle-relative completion. Relative to MAPPO, return increases from 783.75 to **842.26**, recognized completion from 0.440 to **0.497**, and oracle-relative completion from 0.464 to **0.556**. The inference cost stays on the same low scale as MAPPO (+1.41 s cumulative), while remaining ≈8.3×, ≈7.6×, and ≈5.6× faster than GA, CTAS-D, and CBS-TA-PTC respectively.

### Cross-seed stability

| Method | Return | Recognized Completion | Oracle-Relative Completion |
|---|---|---|---|
| FPRL (Ours, 1×10⁶ steps) | 842.26 ± 21.6 | **0.497 ± 0.009** | **0.556 ± 0.010** |
| MAPPO (2×10⁶ steps) | 783.75 ± 37.1 | 0.440 ± 0.032 | 0.464 ± 0.037 |

The seed-level standard deviation of FPRL is roughly one third that of MAPPO, despite MAPPO receiving twice the interaction budget and observing the full, non-sparsified feature set.

### Ablation study — 229-scene fixed benchmark (3 seeds)

| Variant | Completion | CP-SAT Relative | Inference (ms) |
|---|---|---|---|
| **Full** | **0.487 ± 0.006** | **0.545 ± 0.007** | 4.36 |
| w/o GTE | 0.473 ± 0.014 | 0.531 ± 0.014 | **2.15** |
| w/o mask | 0.220 ± 0.090 | 0.251 ± 0.096 | 3.99 |
| w/o commitment | 0.428 ± 0.020 | 0.446 ± 0.025 | 4.10 |
| Self-only graph (param-matched) | 0.430 ± 0.016 | 0.488 ± 0.017 | 3.19 |

**Importance ordering.** Hard action masking dominates (CP-SAT relative completion drops from 0.545 to 0.251, ≈ −54%, with by far the largest seed variance), followed by ordered commitment (≈ −18%). Graph-structure removal has a smaller but consistent effect (w/o GTE −2.6%; parameter-matched self-only −10.5%), confirming that the Full–w/o GTE gap is attributable to graph message passing itself rather than to parameter count.

### Scale robustness — 8 scales × 35 scenes

| Scale | UAVs | Tasks | FPRL | MAPPO | CBS-TA-PTC | GA |
|---|---|---|---|---|---|---|
| s1 | 12 | 24 | **6.17** | 5.89 | 4.31 | 1.89 |
| s2 | 16 | 32 | **7.25** | 6.60 | 5.69 | 1.74 |
| s3 | 20 | 40 | **7.27** | 7.17 | 5.80 | 1.34 |
| s4 | 24 | 48 | **7.63** | 7.46 | 6.43 | 1.49 |
| s5 | 28 | 56 | **9.29** | 9.14 | 8.03 | 1.54 |
| s6 | 32 | 64 | **10.02** | 9.91 | 8.03 | 2.29 |
| s7 | 36 | 72 | 11.08 | **11.14** | 8.69 | 2.51 |
| s8 | 40 | 80 | **14.32** | 14.06 | 8.17 | 2.43 |

Values are mean completed tasks. FPRL leads on six of eight scales and at the largest scale, using only half of MAPPO's training budget; the gap over the non-learned baselines widens with scale.

---

## 🔬 Reproduce the Paper Results

### Main comparison table

```bash
python scripts/make_main_table.py
```

Reads `results/eval_ablation_final_bottleneck/full_seed{1,2,3}/episode_metrics.csv` (Ours), `results/eval_mappo_fair_baseline/full_seed{1,2,3}/episode_metrics.csv` (MAPPO), and `results/baseline_online_v4/{ga,cbs_ta_ptc,ctas_d}/episode_metrics.csv` (baselines), then reports mean / median / IQM / optimality gap with confidence intervals via `rliable`. Outputs are written to `results/analysis/`.

> **Protocol note.** The packaged online-baseline artifacts (`results/baseline_online_v4/`) were collected under the ablation-pipeline configuration, so the table this script prints has an Ours row of `0.545` (the *Full* variant of the ablation table above) rather than the strict-run `0.556` of the main comparison table. The within-table ordering is unchanged; the strict protocol numbers reported in the paper come from the dedicated strict re-runs.

### Figures

```bash
python scripts/regenerate_b_figure.py       # metrics vs. difficulty (six bands)
python scripts/make_main_figures.py         # main-metric figures
python scripts/make_convergence_figure.py   # training convergence
```

Pre-generated copies live in `figures/`. The convergence script requires training logs generated by the training commands below; these logs are not bundled. Other optional `plot_*.py` utilities may require explicit input paths or training artifacts: consult their `--help` before running them.

### Ablation study (re-train all 5 variants × 3 seeds)

This is the long job (≈36 h on one GPU):

```bash
python run_ours_ablation.py \
  --workspace . \
  --seeds 1,2,3 \
  --variants full,wo_rgat,self_only_graph,wo_mask,wo_commitment \
  --save_root checkpoints/ablation_final_bottleneck \
  --eval_output_root results/eval_ablation_final_bottleneck \
  --log_root results/eval_ablation_final_bottleneck/logs \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --oracle_metrics_csv data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1/oracle_episode_metrics.csv \
  --policy_decode greedy --policy_explore_eps 0.0 \
  --require_strict_protocol \
  --skip_completed_train --resume_existing_train --skip_existing \
  --train_extra_args \
    --env_backend subproc --total_steps 1000000 --rollout_steps 768 \
    --num_envs 12 --update_epochs 3 --minibatch_size 384 --env_max_steps 800 \
    --env_difficulty_min 0.2 --env_difficulty_max 0.8 --enforce_env_difficulty \
    --env_difficulty_bins 0.20:0.25,0.25:0.30,0.30:0.35,0.35:0.40,0.40:0.45,0.45:0.50,0.50:0.55,0.55:0.60,0.60:0.65,0.65:0.70,0.70:0.75,0.75:0.80 \
    --curriculum_steps 0 --env_hard_focus_prob 0.0 --env_difficulty_max_attempts 160 \
    --no_auto_tune_runtime
```

The packaged ablation artifacts are already in `results/eval_ablation_final_bottleneck/`.

### Fair MAPPO baseline

```bash
python run_mappo_ablation.py \
  --workspace . \
  --seeds 1,2,3 --variants full \
  --paper_fair_mappo --paper_fair_protocol \
  --save_root checkpoints/mappo_fair_baseline \
  --eval_output_root results/eval_mappo_fair_baseline \
  --log_root results/eval_mappo_fair_baseline/logs \
  --fixed_set_path data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1 \
  --oracle_metrics_csv data/MRTAFixedTestSet_oracle229_d020_080_balanced_v1/oracle_episode_metrics.csv \
  --policy_decode greedy --policy_explore_eps 0.0 \
  --skip_completed_train --resume_existing_train --skip_existing \
  --train_extra_args \
    --no_sparse_dependency_features \
    --env_backend subproc --total_steps 2000000 --rollout_steps 768 \
    --num_envs 12 --update_epochs 3 --minibatch_size 384 --env_max_steps 800 \
    --env_difficulty_min 0.2 --env_difficulty_max 0.8 --enforce_env_difficulty \
    --no_auto_tune_runtime
```

### Scale suite (s1 … s8)

```bash
python run_scale_full_suite.py --help     # presets, orchestration, and evaluation options
python plot_scale_compare.py              # scale comparison figure from results/scale_suite
```

### Mechanism diagnostics

```bash
python diagnose_structure_mechanisms.py --help
```

Reference artifacts are in `results/mechanism_diagnostics/`.

---

## 📁 Project Structure

```
FPRL/
├── train_mrta.py                     # PPO / CTDE trainer (graph-transformer policy)
├── train_config.py                   # TrainConfig dataclass (all training hyper-parameters)
├── mrta_policy.py                    # Policy network: graph transformer encoder + actor/critic heads
├── torch_runtime.py                  # Device selection with safe CUDA -> CPU fallback
├── evaluate_mrta.py                  # Evaluation driver (policy + GA / CBS-TA-PTC / CTAS-D)
├── evaluate_cpsat_oracle.py          # CP-SAT oracle audit for a fixed test set
├── generate_fixed_mrta_testset.py    # Fixed-benchmark generator
├── run_ours_ablation.py              # Orchestrator: train + eval the ablation variants
├── run_mappo_ablation.py             # Orchestrator: train + eval the fair MAPPO baseline
├── run_scale_full_suite.py           # Orchestrator: scale-robustness suite (s1..s8)
├── diagnose_structure_mechanisms.py  # Structure / mechanism diagnostics
├── plot_ablation_importance.py       # \
├── plot_ablation_training.py         #  |
├── plot_bc_from_eval_artifacts.py    #  |  individual figure scripts
├── plot_loss_step_convergence.py     #  |
├── plot_scale_compare.py             #  |
├── plot_seed_stability.py            #  |
├── plot_value_loss_with_mappo.py     # /
├── requirements.txt
├── LICENSE                           # Apache License 2.0
├── README.md
│
├── env/                              # Environment package
│   ├── mrta_env.py                   #   core dynamic heterogeneous MRTA environment
│   ├── mrta_config.py                #   MRTAConfig dataclass (env hyper-parameters)
│   ├── fixed_set_loader.py           #   fixed-benchmark loader
│   ├── subproc_vec_env.py            #   subprocess vectorized env for parallel rollouts
│   ├── demo.gif / drone.png          #   visualization assets
│   └── __init__.py
│
├── baseline/                         # Traditional / online baselines
│   ├── legacy_adapter.py             #   unified adapter used by evaluate_mrta.py
│   ├── original_baseline_adapter.py  #   planning core shared by all baselines
│   ├── ga_assignment.py              #   genetic-algorithm assignment
│   ├── cbs_ta_ptc_assignment.py      #   CBS task assignment + precedence/temporal constraints
│   ├── cbs_ta_ptc_2024.py            #   AAMAS-2024 CBS variant
│   ├── cbs_ta_ptc_lowlevel.py        #   low-level single-robot planners
│   ├── cbs_ta_ptc_constraints.py     #   constraint bookkeeping
│   ├── ctasd_compat.py               #   original CTAS-D TSP solver loader (+ fallback)
│   └── CTAS-D.py                     #   reference CTAS-D implementation
│
├── scripts/                          # Paper table / figure regeneration
│   ├── make_main_table.py            #   main comparison table (uses rliable)
│   ├── make_main_figures.py          #   main-metric figures
│   ├── make_convergence_figure.py    #   training-convergence figure
│   └── regenerate_b_figure.py        #   paper-specific metrics-vs-difficulty layout
│
├── data/                             # Benchmarks (read-only)
│   ├── MRTAFixedTestSet_oracle229_d020_080_balanced_v1/
│   │   ├── env_0 ... env_228/        #   229 scenes (one config.json each)
│   │   ├── oracle_episode_metrics.csv
│   │   ├── overview.csv
│   │   └── selection_manifest.json
│   └── scale_suite_fixed_sets/s1 ... s8/
│
├── results/                          # Raw evaluation artifacts used in the paper
│   ├── baseline_online_v4/{ga,cbs_ta_ptc,ctas_d}/
│   ├── eval_ablation_final_bottleneck/{full,wo_rgat,self_only_graph,wo_mask,wo_commitment}_seed{1,2,3}/
│   ├── eval_mappo_fair_baseline/full_seed{1,2,3}/
│   ├── scale_suite/{eval,oracle,plots}/
│   ├── mechanism_diagnostics/
│   └── analysis/                     # generated by scripts/make_main_table.py
│       ├── main_table_rliable.md
│       └── performance_profile_rliable.png
│
├── figures/                          # Pre-generated paper figures
│   ├── archi.png                     #   framework overview
│   ├── architecture_diagram_editable.pptx   # editable framework diagram
│   ├── B_metrics_vs_difficulty.png   #   metrics vs. difficulty (six bands)
│   ├── ablation_analysis.png         #   ablation summary
│   ├── value_loss_convergence.png    #   value-loss convergence
│   ├── loss_step_convergence.png     #   loss / step convergence
│   └── suite_summary.json            #   scale-suite numeric summary
│
└── checkpoints/                      # (NOT bundled) trained weights, see Model Zoo
```

> The paper protocol uses `--minibatch_size 384` explicitly (see the ablation and MAPPO commands above); the `TrainConfig` default in `train_config.py` is `1536`, so always pass the flag when reproducing the reported numbers.

---

## 🧩 Customization

### Adding an ablation variant

1. Add the flag to `train_mrta.py` (for example `--wo_my_module`), and thread it through `TrainConfig` in `train_config.py`.
2. Gate the corresponding module in `mrta_policy.py` (encoder / decoding / masking path).
3. Add the variant name to the `--variants` list of `run_ours_ablation.py`; the orchestrator trains and evaluates it automatically.

> **Important:** any modification of the observation layout, the feature dimensions, or the masking rule invalidates existing checkpoints — `evaluate_mrta.py` will refuse to load an incompatible one and prints the mismatching keys.

### Changing the scenario

Environment knobs are CLI flags on both `train_mrta.py` and `evaluate_mrta.py`, e.g. `--env_num_agents`, `--env_max_tasks`, `--env_init_tasks_min/max`, `--env_dynamic_arrival_every`, `--env_difficulty_min/max`. For evaluation, the scene configuration stored in a fixed set's `config.json` always takes precedence, unless `--fixed_set_max_steps` overrides the horizon.

### Adding a new baseline

Implement a class exposing `plan(env)`, `act(env)` / `act_from_plan(env, plan)`, `reset_episode()`, and `get_episode_stats()`, then register it in `make_legacy_baseline_adapter()` inside `evaluate_mrta.py` and extend the `--baseline_method` choices.

---

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{wan2026fprl,
  title   = {Graph-Transformer-Aided Reinforcement Learning for Dynamic Heterogeneous
             Multi-UAV Task Allocation with Temporal Dependencies},
  author  = {Wan, Zumin and Chen, Tao and Ren, Bangbang and Li, Ruobin},
  journal = {Drones},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This work was carried out at the National Key Laboratory of Information Systems Engineering, National University of Defense Technology. We thank the authors of the CBS-OTA reference implementation for releasing their code, which serves as the planning-based baseline, and the developers of `PyTorch`, `OR-Tools` (CP-SAT), and `rliable` for the open tooling that made this study possible.

---

## 📧 Contact

- **Tao Chen** (corresponding author) — chentao@nudt.edu.cn
- **Zumin Wan** — wanzumin@nudt.edu.cn

National University of Defense Technology (NUDT), Changsha, China

For bug reports and feature requests, please open an issue with the exact command line, the Python/PyTorch versions, and the relevant log excerpt.

---

<div align="center">

**⭐ If this project helps your research, please give us a star! ⭐**

</div>
