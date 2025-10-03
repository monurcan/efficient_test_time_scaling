export CUDA_VISIBLE_DEVICES=0,1

module purge
module load gcc
module load python3/3.10.12
module load cuda/12.5
module load latex

source .venv/bin/activate

# export CACHE_PATH=/dtu/blackhole/00/215456/.CACHE
export CACHE_PATH=/work3/monka/.CACHE

export HF_HOME=$CACHE_PATH
export TRANSFORMERS_CACHE=$CACHE_PATH
export DATASETS_CACHE=$CACHE_PATH
export TORCH_HOME=$CACHE_PATH
export TORCH_CACHEDIR=$CACHE_PATH
export CUDA_CACHE_PATH=$CACHE_PATH

# Weights and Biases cache and data
export WANDB_CACHE_DIR=$CACHE_PATH
export WANDB_DATA_DIR=$CACHE_PATH

# General Python cache directory for .pyc files
export PYTHONPYCACHEPREFIX=$CACHE_PATH
export TMPDIR=$CACHE_PATH/tmp

# Datasets cache directory 
export LMUData=$CACHE_PATH/LMUData

export NO_ALBUMENTATIONS_UPDATE=1

# Huggingface login
hfkey_file_path="/zhome/88/8/215456/hf_key.txt"
export HF_TOKEN=$(<"$hfkey_file_path")
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# If there is no argument, run Jupyter Notebook
if [ $# -eq 0 ]; then
    python -m notebook --ip 0.0.0.0 --no-browser --port=8080 --allow-root
else
    echo "Jupyter not running"
fi