#!/bin/bash
# NOTE: SLURM does not expand shell variables inside #SBATCH directives, so the
# account/partition/log paths below must be edited directly (or overridden on
# the command line, e.g. `sbatch --account=... --partition=...`) -- see
# README > Configuration.
#SBATCH -A ISCRA_PROJECT_NAME
#SBATCH --job-name=dromedario_sft
#SBATCH --output=/leonardo_scratch/large/userexternal/%u/dromedario/logs/single-node_sft_%j.out.log
#SBATCH --error=/leonardo_scratch/large/userexternal/%u/dromedario/logs/single-node_sft_%j.err.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --qos=normal
 
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================="

# --- Load CINECA Modules and Environment ---
module purge
module load cuda/12.2
module load gcc

unset HF_HUB_OFFLINE
export HF_HUB_OFFLINE=1

# SCRATCH_ROOT is the one thing to change to run this on a different cluster/filesystem.
# Set it in your environment before sbatching, otherwise it defaults to the CINECA Leonardo path.
export SCRATCH_ROOT="${SCRATCH_ROOT:-/leonardo_scratch/large/userexternal/$USER}"
export PROJECT_DIR="$SCRATCH_ROOT/dromedario"
export VENV_DIR="$PROJECT_DIR/llama_env"
cd $PROJECT_DIR

export TMPDIR="$SCRATCH_ROOT/tmp"
mkdir -p $TMPDIR

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$SCRATCH_ROOT/.hf_cache"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
mkdir -p $TORCH_EXTENSIONS_DIR
export TORCH_COMPILE_DISABLE=1
export PIP_CACHE_DIR="$TMPDIR/pip_cache"
export UV_CACHE_DIR="$TMPDIR/uv_cache"
export TIKTOKEN_CACHE_DIR="$SCRATCH_ROOT/tiktoken"
mkdir -p $PIP_CACHE_DIR $UV_CACHE_DIR $TIKTOKEN_CACHE_DIR
# export HF_TOKEN="_"
export PYTHONPATH=$PROJECT_DIR
export WANDB_MODE=offline


# Check if the required models are present in the Hugging Face cache before starting training
models=(
    "meta-llama/Llama-3.1-8B"
    "sapienzanlp/Minerva-7B-base-v1.0"
)
for model in "${models[@]}"; do
    if [ -d "$HF_HOME/hub/models--${model//\//--}" ]; then
        echo "Model $model found in cache"
    else
        echo "Model $model NOT found in cache!"; exit 1
    fi
done

 
# Activate your custom environment where LLaMA Factory is installed
source $VENV_DIR/bin/activate
export VIRTUAL_ENV=$VENV_DIR
export PATH="$VENV_DIR/bin:$PATH"

# --- Local Settings ---
export FORCE_TORCHRUN=1                    # Still needed to invoke torchrun
export GPUS_PER_NODE=4                     # Set this to match your --gres=gpu:X allocation
export OMP_NUM_THREADS=8                   # how many CPU threads each training process can use for basic operations

# --- NCCL Tuning for HPC Fabrics ---
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                   # Force InfiniBand usage over slow TCP
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "========================================================================="
echo "LAUNCHING SINGLE-NODE JOB"
echo "========================================================================="

# --- Execute Training ---
# (this repo is expected to be cloned directly into $PROJECT_DIR -- see README > Configuration)
llamafactory-cli train "$PROJECT_DIR/_configs/llama3___control.yaml"
