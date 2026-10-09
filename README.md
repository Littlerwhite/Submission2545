# Reproducibility Package

Code for the paper on DeepCFR-based dynamic honeypot deployment on attack graphs.

## Requirements

- Ubuntu 22.04 or 24.04 (tested)
- ~10 GB disk space
- 4+ CPU cores, 8+ GB RAM recommended
- Internet access (GitHub and PyPI)

## Quick start

    bash setup.sh

This script will:

1. Install system dependencies (CMake, G++, Python 3)
2. Clone and compile OpenSpiel v2.0.1 (~15 min)
3. Create a Python virtual environment with PyTorch (CPU)
4. Register our custom attack-graph games
5. Run smoke tests to verify the installation

**Total time: 20-30 minutes** (compilation dominates).

## Running experiments

After `setup.sh` completes:

    cd ~/open_spiel
    source venv/bin/activate

### Quick sanity check (30 seconds)

    python3 run_experiment.py --game attack_graph_10 --algo sdcfr --seed 42 --iterations 50

Expected: prints hit rates for 4 attacker types.

### Main experiments

| Experiment | Script | Time |
| :--- | :--- | :--- |
| 10-node, 3 algorithms, 5 seeds | bash run_seeds_simple.sh | ~10 min |
| 100-node ablation | bash run_ablation_100.sh | ~20 min |
| Exact and classical baselines | python3 run_baselines_10.py | ~30 min |

### Regenerating figures

    python3 plot_topologies.py
    python3 plot_ablation.py
    python3 plot_budget.py
    python3 plot_radar.py
    python3 plot_time.py

Output: PDF and PNG in figures/.

## Directory structure

    .
    |-- setup.sh                       # One-click installation
    |-- requirements.txt               # Python dependencies
    |-- attack_graph_scalable.py       # Custom OpenSpiel game
    |-- algorithms/
    |   |-- deep_cfr_dueling.py        # Dueling DeepCFR
    |   |-- single_deep_cfr.py         # SDCFR
    |   `-- dueling_mlp.py             # Dueling network
    |-- run_experiment.py              # Experiment driver
    |-- run_seeds_simple.sh            # 5-seed batch
    |-- run_ablation.sh                # G10 ablation
    |-- run_ablation_100.sh            # G100 ablation
    |-- run_baselines_10.py            # Baselines
    |-- plot_*.py                      # Figure generation
    |-- summarize_results.py           # Result aggregation
    `-- compute_kappa.py               # Min vertex cut analysis

## Troubleshooting

**ModuleNotFoundError: No module named pyspiel**

Reload the shell or set PYTHONPATH manually:

    export PYTHONPATH=$HOME/open_spiel/open_spiel/build/python:$HOME/open_spiel:$PYTHONPATH

**Compilation fails near 97 percent**

Memory issue. Try `make -j2 pyspiel` or add swap:

    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

**GitHub connection fails**

Use gitee mirrors for OpenSpiel dependencies.

## License

Code released for academic reproducibility.
