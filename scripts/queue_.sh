#!/bin/sh

# q = gpua100, gpuh100, p1
# https://www.hpc.dtu.dk/?page_id=2759
# n = 4 * gpucount
# walltime is 72 for p1, 24 for others
# CUDA_VISIBLE_DEVICES also change it!

#BSUB -q p1

#BSUB -W 24:00

#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"

#BSUB -J exp_83_newdatasets_finalized_ovis1000_img
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"

#BSUB -o benchmark_queue_outputs/onur%J.out
#BSUB -e benchmark_queue_outputs/onur%J.err

#BSUB -u monka@dtu.dk
#BSUB -B
#BSUB -N

export WORK_DIR="/work3/monka/tta-vlm-new"
cd "$WORK_DIR"
source scripts/prepare_env.sh nojp
#bash scripts/local_judge.sh start
exp_config_path="${WORK_DIR}/benchmark_configs/exp_83_newdatasets_finalized_ovis1000_img.json"

####################
# benchmark.sh
use_openai=False # True, False, "Local" Note: True is too expensive!, For local judge, use local_judge.sh script.
# export CUDA_VISIBLE_DEVICES=0,1
export AUTO_SPLIT=0
export SUBSET_LEN=1000
export USE_COT=1
export TOKENIZERS_PARALLELISM=false
export DIST_TIMEOUT=99999999999
export UNSLOTH_DISABLE_FAST_GENERATION="1"

exp_config_stem=$(basename "$exp_config_path" .json)
workdir="${WORK_DIR}/benchmark_results/n_samples_${SUBSET_LEN}/${exp_config_stem}/"

# if [ -d "$workdir" ]; then
#     read -p "The directory '$workdir' already exists. Do you want to delete it? (y/n): " confirm
#     if [ "$confirm" = "y" ]; then
#         rm -r "$workdir"
#     fi
# fi

mkdir -p "$workdir"
cp "$exp_config_path" "$workdir"

if [ "$use_openai" = True ]; then
    openai_file_path="/zhome/88/8/215456/openai_key.txt"
    export OPENAI_API_KEY=$(<"$openai_file_path")
elif [ "$use_openai" = "Local" ]; then
    export OPENAI_API_KEY="sk-123456"
    export OPENAI_API_BASE="http://0.0.0.0:23333/v1/chat/completions"
    export LOCAL_LLM="internlm/internlm2_5-7b-chat"

    for _ in {1..3}; do echo "================================================================"; done
    echo "Local judge is enabled. Make sure you have the local judge running."
    for _ in {1..3}; do echo "================================================================"; done
fi

export PYTHONPATH=$PWD:$PYTHONPATH
number_of_gpus=$(($(grep -o "," <<<"$CUDA_VISIBLE_DEVICES" | wc -l) + 1))
echo "Number of GPUs: $number_of_gpus"
if [ "$number_of_gpus" -gt 1 ]; then
    if [ "$AUTO_SPLIT" = 1 ]; then
        number_of_gpus=1
    fi
    torchrun --standalone --nproc-per-node=$number_of_gpus --master-port 29555 run.py --work-dir "$workdir" --verbose --reuse --config "$exp_config_path"
else
    python run.py --work-dir "$workdir" --verbose --reuse --config "$exp_config_path"
fi
