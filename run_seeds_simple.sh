#!/bin/bash
cd "$(dirname "$0")"
source ~/open_spiel/venv/bin/activate

for seed in 42 123 456 789 1010; do
    echo "===== Simple | Seed $seed ====="
    python3 run_experiment.py \
        --game attack_graph_100 \
        --algo deepcfr \
        --seed $seed \
        --iterations 150 \
        --output_dir results
done
