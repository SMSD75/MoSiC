module load gnu12
HOSTNAME=$(hostname)

# Set the data directory based on the hostname
if [ "$HOSTNAME" = "ivi-cn031" ]; then
    export DATA_PREFIX="/nvmestore/ssalehi"
elif [ "$HOSTNAME" = "ivi-cn023" ]; then
    export DATA_PREFIX="/ssdstore/ssalehi"
elif [ "$HOSTNAME" = "ivi-cn028" ]; then
    export DATA_PREFIX="/ssdstore/ssalehi"
else
    export DATA_PREFIX="/scratch/ssalehi/"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Create directories if they don't exist
mkdir -p $DATA_PREFIX/TRITON_CACHE
mkdir -p $DATA_PREFIX/wandb_mosic
mkdir -p $DATA_PREFIX/.cache

export TRITON_CACHE_DIR=$DATA_PREFIX/TRITON_CACHE
export WANDB_DIR=$DATA_PREFIX/wandb_mosic/
export HF_HOME=$DATA_PREFIX/.cache/

# Define master address and port
export MASTER_ADDR='localhost'
export MASTER_PORT='12356'

# Add the current directory to PYTHONPATH to ensure imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

# train_split_list=(
#     "trainaug_128_1293.txt"
#     "trainaug_128_42.txt"
#     "trainaug_128_400.txt"
#     "trainaug_128_783.txt"
#     "trainaug_128_4019.txt"
#     "trainaug_64_400.txt"
#     "trainaug_64_783.txt"
#     "trainaug_64_1293.txt"
#     "trainaug_64_42.txt"
#     "trainaug_64_4019.txt"
#     "trainaug_8_4019.txt"
#     "trainaug_8_1293.txt"
#     "trainaug_8_42.txt"
#     "trainaug_8_400.txt"
# )

train_split_list=(
    # "training_128_1293.txt"
    # "training_128_42.txt"
    # "training_128_400.txt"
    # "training_128_783.txt"
    # "training_128_4019.txt"
    # "training_64_400.txt"
    # "training_64_783.txt"
    # "training_64_1293.txt"
    # "training_64_42.txt"
    # "training_64_4019.txt"
    # "training_8_4019.txt"
    # "training_8_1293.txt"
    # "training_8_42.txt"
    # "training_8_400.txt"
)

for i in "${!train_split_list[@]}"; do
    train_split=${train_split_list[$i]}
    output_file="hb/hummingbird_MoSiC_dinov2-l_ade20k${train_split%.*}.log"
    
    torchrun --nproc_per_node=3 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    evaluation/evaluation.py \
    --model mosic_dinov2-l \
    --input-size 518 \
    --batch-size 24 \
    --embeddings-size 1024 \
    --patch-size 14 \
    --memory-size 10240000 \
    --num-workers 2 \
    --dataset ade20k \
    --data-dir $DATA_PREFIX/ADE20k/ADEChallengeData2016 \
    --train-split ${train_split%.*} \
    2>&1 | tee $output_file
done
