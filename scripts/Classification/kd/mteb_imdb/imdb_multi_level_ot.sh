#! /bin/bash
GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=/content/LLM2Vec_Distillation
CKPT_NAME="bert"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_NAME="LLM2Vec"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"
# data
DATA_DIR="${BASE_PATH}/data/imdb/"
NUM_LABELS=2
# task
TASK="multi_level_ot"
# hp
BATCH_SIZE=4
LR=0.00001
GRAD_ACC=1
EVAL_BATCH_SIZE=4
EPOCH=3
KD_RATE=0.5
KD_TEMP=2.0
# length
MAX_LENGTH=512

# runtime
PRECISION="bf16"
CRITERION="multi_level_ot"
KD_OBJ="forward_kl"  # [forward_kl, reverse_kl, js_divergence, skewed_forward_kl, skewed_reverse_kl, adaptive_kl]
CONFIG="${KD_OBJ}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${KD_RATE}__kd^temp=${KD_TEMP}__tea^temp=${TEA_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
# seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --num-labels ${NUM_LABELS}"
# task
OPTS+=" --task ${TASK}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-eval"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_test.json"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/Classification/distillation.py ${OPTS}"

${CMD} \
>> ${SAVE_PATH}/train.log 2>&1 &
