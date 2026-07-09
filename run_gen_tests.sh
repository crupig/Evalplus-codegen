DEVICE=$1
MODEL_PATH=$2
DATASET=mbpp
IFS='/' read -r MODEL_FAMILY MODEL_NAME <<< "$MODEL_PATH"

IDS_FILE="../../constants/ids_train_val_test.json"

CUDA_VISIBLE_DEVICES=$DEVICE evalplus.evaluate \
    --model $MODEL_PATH \
    --dataset $DATASET \
    --backend vllm \
    --temperature 1.0 \
    --n_samples 5 \
    --dtype bfloat16 \
    --root ./test-generations/${MODEL_FAMILY}--${MODEL_NAME} \
    --generate_tests \
    --execute_tests no \
    --all_ids_dict $IDS_FILE \
    --split all # train, val, test, or all
