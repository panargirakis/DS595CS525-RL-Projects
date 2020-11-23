#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem 16000
#SBATCH --gres=gpu:1
#SBATCH -C K20
#SBATCH -o log.txt

# Load any needed environment modules
module load python/gcc-8.2.0/3.7.2
module load cuda92/blas/9.2.88
module load cuda92/fft/9.2.88
module load cuda92/nsight/9.2.88
module load cuda92/profiler/9.2.88
module load cuda92/toolkit/9.2.88

# Set python virtual env
source /home/pargyrakis/DS595CS525-RL-Projects/venv/bin/activate

# Execute my commands
cd /home/pargyrakis/DS595CS525-RL-Projects/Project3
pip install -r requirement.txt
pip install torch==1.7.0+cu92 torchvision==0.8.1+cu92 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
python main.py --train_dqn
