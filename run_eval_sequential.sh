BASE_GENERATIONS_PATH=$1   # e.g. ./generations/google--gemma-3-1b-it/google--gemma-3-1b-it-Evalplus-part

for i in {0..9}; do
    # Replace trailing _<digit>.jsonl with _i.jsonl
    GENERATIONS_PATH="${BASE_GENERATIONS_PATH}${i}.jsonl"
    bash run_eval.sh $GENERATIONS_PATH
done
