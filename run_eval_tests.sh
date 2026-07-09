GENERATIONS_PATH=$1
DATASET=mbpp

OUTPUT_PATH="${GENERATIONS_PATH/test-4-execution/test-execution}"
OUTPUT_PATH="${OUTPUT_PATH/\.jsonl/\.eval_results\.json}"
echo "OUTPUT PATH: $OUTPUT_PATH"

evalplus.evaluate \
    --samples $GENERATIONS_PATH \
    --dataset $DATASET \
    --output_file $OUTPUT_PATH \
    --execute_tests yes \
    --parallel 8