#!/bin/bash
#SBATCH -D /users/adch204/quaternion_unet       # Working directory
#SBATCH --job-name=qunet_train                  # Job name
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=benny.collins@city.ac.uk
#SBATCH --output=qunet_train_%j.out             # Standard output and error log [%j is replaced with the jobid]
#SBATCH --error=qunet_train_%j.error
#SBATCH --time=48:00:00                         # Time limit hrs:min:sec
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB                              # Expected memory usage (0 means use all available memory)
#SBATCH --ntasks=6
#SBATCH --partition=gengpu

# Enable modules
source /opt/flight/etc/setup.sh
flight env activate gridware
# Run .bashrc
source /users/adch204/.bashrc

# Unload all currently loaded modules (clean environment)
module purge 
# Load modules required
module load python/3.7.12
module load gnu
module load libs/nvidia-cuda/11.2.0/bin
module load cudnn/8.5.0
module load ffmpeg

# Activate conda environment
source qunet_env/bin/activate

# Run the script
python main.py --train_standard_unet False
