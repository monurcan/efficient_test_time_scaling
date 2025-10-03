#!/bin/sh

# q = gpua100, gpuh100, p1
# https://www.hpc.dtu.dk/?page_id=2759
# n = 4 * gpucount
# walltime is 72 for p1, 24 for others

#BSUB -q gpuh100
#BSUB -W 24:00

#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"

#BSUB -J onur
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"

#BSUB -o benchmark_queue_outputs/onur%J.out
#BSUB -e benchmark_queue_outputs/onur%J.err

#BSUB -u monka@dtu.dk
#BSUB -B
#BSUB -N

cd /work3/monka/tta-vlm-new
source scripts/prepare_env.sh nojp
#bash scripts/local_judge.sh start
bash scripts/benchmark.sh benchmark_configs/exp_81_8augcount.json
