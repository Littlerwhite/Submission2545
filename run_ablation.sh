#!/bin/bash
cd ~/open_spiel
source ~/open_spiel/venv/bin/activate

for game in attack_graph_10 attack_graph_10_nodynamic attack_graph_10_nohidden; do
    for seed in 42 123 456; do
        echo "=== $game | seed $seed ==="
        python3 run_experiment.py \
            --game $game --algo sdcfr --seed $seed \
            --iterations 200 --output_dir results_ablation
    done
done
