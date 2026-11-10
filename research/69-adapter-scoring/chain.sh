#!/usr/bin/env bash
# #69: B then D step 1000 and B step 1500, scored with merged weights in vLLM and with HF + PEFT.
set -euo pipefail
cd /home/joao/orca/workspaces/ml/ml
S=/home/joao/orca/workspaces/ml/w69/research/69-adapter-scoring
O=runs/69-adapter-scoring
BD=runs/train-opd-from-sft-teacher-r16-b4x4-n52424-gl-s0-1500-r16-b4x4-n54561-s0/checkpoint-1000
B=runs/train-sft-teacher-r16-b4x4-n52424-gl-s0/checkpoint-1500
mkdir -p $O
for pair in "bd:$BD" "b:$B"; do
  name=${pair%%:*}; ck=${pair#*:}
  echo "== $name merge $(date +%T)"
  uv run --extra inference python $S/merge_adapter.py $ck $O/merged-$name
  echo "== $name fc-eval merged $(date +%T)"
  uv run --extra inference fc-eval --model $O/merged-$name --split validation --out $O/eval-validation-merged-$name
  echo "== $name hf $(date +%T)"
  uv run --extra inference python $S/hf_answers.py $ck $O/hf-$name.jsonl
done
echo "== done $(date +%T)"
