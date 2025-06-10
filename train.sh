module load gnu12
HOSTNAME=$(hostname)

# Set the data directory based on the hostname
if [ "$HOSTNAME" = "ivi-cn031" ]; then
    export DATA_PREFIX="/nvmestore/ssalehi"
elif [ "$HOSTNAME" = "ivi-cn030" ]; then
    export DATA_PREFIX="/nvmestore/ssalehi"
elif [ "$HOSTNAME" = "ivi-cn023" ]; then
    export DATA_PREFIX="/ssdstore/ssalehi"
elif [ "$HOSTNAME" = "ivi-cn028" ]; then
    export DATA_PREFIX="/ssdstore/ssalehi"
else
    export DATA_PREFIX="/scratch/ssalehi/"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
# Create directories if they don't exist
mkdir -p $DATA_PREFIX/TRITON_CACHE
mkdir -p $DATA_PREFIX/wandb_mosic
mkdir -p $DATA_PREFIX/.cache

export TRITON_CACHE_DIR=$DATA_PREFIX/TRITON_CACHE
export WANDB_DIR=$DATA_PREFIX/wandb_mosic/
export HF_HOME=$DATA_PREFIX/.cache/

# Define master address and port
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'

# Add the current directory to PYTHONPATH to ensure imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

torchrun --nproc_per_node=8 \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
experiments/exp_mosic_multigpu.py \
--batch_size 64 \
--frame_sampling_mode regular \
--regular_step 6 \
--num_clip_frames 12 \
--num_clips 1 \
--num_epochs 8 \
--num_prototypes 100 \
--feature_upsampling nearest \
--num_workers 8 \
--model_type dinov2-s \
--explaination mixed_dataset_debug \
--dataset ytvos \
--mask_ratio 0 \
--grid_size 16 \
--crop_scale 0.4 \
--wandb_mode online \
--use_EMA_teacher True \
--teacher_feature_upsampling nearest \
--save_dir $DATA_PREFIX/MoSiC_models