#!/bin/bash
# Launch 10-fold CV for GCN-only training on 3 GPUs in parallel
# Runs 3 folds at a time (one per GPU), waits for batch completion

set -e
PROJECT_DIR="/home/lyf/projects/TCR-ECHO"
PYTHON="/home/lyf/miniconda3/envs/tcr-echo-5090/bin/python"
N_FOLDS=10
N_GPUS=3
LR=1e-4

cd "$PROJECT_DIR"

echo "=== 10-Fold CV on $N_GPUS GPUs ==="
echo "Start: $(date)"

RESULTS_DIR="$PROJECT_DIR/runs/gcn_only"
mkdir -p "$RESULTS_DIR"

# Batch-launch folds: 3 at a time
for ((batch_start=0; batch_start<N_FOLDS; batch_start+=N_GPUS)); do
    batch_pids=()
    for ((gpu=0; gpu<N_GPUS && batch_start+gpu<N_FOLDS; gpu++)); do
        fold=$((batch_start + gpu))
        echo "[$(date)] Launching fold $fold on GPU $gpu (LR=$LR)"
        mkdir -p "$RESULTS_DIR/fold_${fold}"
        nohup env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
            "$PYTHON" -u gcn_only_train.py \
            --fold "$fold" --n-folds "$N_FOLDS" --gpu "$gpu" --lr "$LR" \
            > "$RESULTS_DIR/fold_${fold}/stdout.log" 2>&1 &
        batch_pids+=($!)
        sleep 2  # stagger GPU memory allocation
    done
    echo "[$(date)] Batch running: folds $batch_start-$((batch_start + ${#batch_pids[@]} - 1)), PIDs: ${batch_pids[*]}"
    for pid in "${batch_pids[@]}"; do
        wait "$pid"
        echo "[$(date)] PID $pid finished (exit code $?)"
    done
done

echo ""
echo "=== All folds complete ==="
echo "End: $(date)"

# Aggregate results
echo ""
echo "=== Aggregate Results ==="
for fold in $(seq 0 $((N_FOLDS - 1))); do
    log="$RESULTS_DIR/fold_${fold}/training.log"
    if [ -f "$log" ]; then
        best=$(grep "Best val AUC:" "$log" | tail -1 | awk '{print $NF}')
        test_auc=$(grep "Test AUC:" "$log" | tail -1 | awk '{print $NF}')
        echo "Fold $fold: val AUC=$best, test AUC=$test_auc"
    else
        echo "Fold $fold: MISSING"
    fi
done
