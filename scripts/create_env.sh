export CUDA_VISIBLE_DEVICES=0,1

module purge
module load gcc
module load python3/3.10.12
module load cuda/12.5
module load latex

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-deps
pip install -e . --no-deps
