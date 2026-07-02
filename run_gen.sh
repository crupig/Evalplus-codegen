DEVICE=$1
MODEL_PATH=$2
DATASET=mbpp
IFS='/' read -r MODEL_FAMILY MODEL_NAME <<< "$MODEL_PATH"

IDS_FILE="../../constants/ids_train_val_test.json"

CUDA_VISIBLE_DEVICES=$DEVICE evalplus.evaluate \
    --model $MODEL_PATH \
    --dataset $DATASET \
    --backend hf \
    --temperature 1.0 \
    --n_samples 1 \
    --dtype bfloat16 \
    --bs 1 \
    --execute_tests no \
    --root ./generations/${MODEL_FAMILY}--${MODEL_NAME} \
    --all_ids_dict $IDS_FILE \
    --split test # train, val, test, or all
