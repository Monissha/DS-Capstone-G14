#!/bin/bash
#SBATCH --job-name=phi3_job
#SBATCH --partition=gpu              
#SBATCH --gres=gpu:8                
#SBATCH --nodes=2           #optional                       
#SBATCH --time=10:00:00             
#SBATCH --output=phi3_output_%j.log 
#SBATCH --error=phi3_error_%j.log   

# Load necessary modules for GPU and CUDA
module load cuda
module load cudnn
module load gcc/12.2.0
module load python/3.9.15


# Activate virtual environment
source /scratch/courses0101/username/myenv/bin/activate

# Run the Python script 
python phi3_qlora.py 
