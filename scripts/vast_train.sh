#!/usr/bin/env bash
# Train XLM-R on synthetic v2 (and v1 for comparison) on a rented GPU, predict the internal splits,
# and upload checkpoints + predictions to the HF artifacts repo. The official test is NOT touched.
#
# On the GPU machine:
#   export HF_TOKEN=hf_...            # write token
#   git clone -b data-v2-hybrid https://github.com/Ovkubiky/ua-court-pseudonymizer && cd ua-court-pseudonymizer
#   nohup bash scripts/vast_train.sh > train.log 2>&1 &
#   tail -f train.log
#
# VERSIONS="v2" to skip the v1 run.
set -euo pipefail

REPO=${HF_REPO:-ovkubiky/ua-court-pseudonymizer-artifacts}
VERSIONS=${VERSIONS:-"v2 v1"}
: "${HF_TOKEN:?set HF_TOKEN (write token) first}"

nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q -r requirements.txt "huggingface_hub[cli]>=1.0"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

hf download "$REPO" --repo-type dataset --include "synthetic/*" --local-dir artifacts
mkdir -p data/synthetic data/synthetic_v1
cp artifacts/synthetic/v2/* data/synthetic/
cp artifacts/synthetic/v1/* data/synthetic_v1/
python scripts/verify_synthetic_data.py

for v in $VERSIONS; do
  data=data/synthetic; [ "$v" = v1 ] && data=data/synthetic_v1
  out=baselines/ner_xlm_roberta_$v
  echo "=== train on $v ($data) ==="
  python scripts/ner_baseline.py train --data-dir "$data" --output-dir "$out" \
    --seed 42 --epochs 3 --batch-size 8 --gradient-accumulation 2 --learning-rate 2e-5 \
    --weight-decay 0.01 --max-length 256 --stride 128 --log-every 500

  # predictions on BOTH data versions' internal splits (cross-generator check), raw spans for the hybrid ablation
  for ev in v2 v1; do
    evdata=data/synthetic; [ "$ev" = v1 ] && evdata=data/synthetic_v1
    for split in dev test_a test_b; do
      python scripts/ner_baseline.py infer --split "$split" --data-dir "$evdata" --checkpoint "$out/selected" \
        --output "preds/train_$v/eval_$ev/$split.jsonl" --raw-output "preds/train_$v/eval_$ev/$split.raw.jsonl"
    done
  done

  rm -rf "$out"/epoch-*   # keep only the selected checkpoint (saves ~2 GB of upload)
  hf upload "$REPO" "$out" "models/ner_xlm_roberta_$v" --repo-type dataset --commit-message "XLM-R trained on synthetic $v"
  hf upload "$REPO" "preds/train_$v" "preds/train_$v" --repo-type dataset --commit-message "Predictions of the $v-trained model"
done

echo "ALL DONE - safe to destroy the instance"
