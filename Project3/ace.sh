#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem 16000
#SBATCH --gres=gpu:0
#SBATCH -o log.txt

# Load any needed environment modules
module load python/gcc-8.2.0/3.7.2

# Set python virtual env
source /home/pargyrakis/DS595CS525-RL-Projects/venv/bin/activate

# Execute my commands
cd /home/pargyrakis/DS595CS525-RL-Projects/Project3
pip install -r requirement.txt
python main.py --train_dqn
