#!/bin/sh
set -eu

MODEL="gpt-4o-mini-2024-07-18"
LANG="en"

run_phase() {
  tag="$1"
  i=1
  while [ "$i" -le 3 ]; do
    rm -rf "result_all/result_${LANG}/${MODEL}/"
    python3 generate.py --model="$MODEL" --temperature=0 --top-p=0 --language="$LANG" --num-threads=1
    python3 eval_main.py --model="$MODEL" --category=test_all --language="$LANG"
    cp "score_all/score_${LANG}/result.xlsx" "score_all/score_${LANG}/${tag}_run_${i}.xlsx"
    i=$((i+1))
  done
}

run_phase before
run_phase after
