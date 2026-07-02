MODEL_NAME=$1
DATASET=mbpp

IDS_FILE="../../constants/ids_train_val_test.json"

CUDA_VISIBLE_DEVICES=$DEVICE evalplus.evaluate \
    --model $MODEL_NAME \
    --dataset $DATASET \
    --backend openai \
    --temperature 1.0 \
    --n_samples 10 \
    --bs 10 \
    --execute_tests no \
    --root ./generations/openai--${MODEL_NAME} \
    --all_ids_dict $IDS_FILE \
    --split test # train, val, test, or all
