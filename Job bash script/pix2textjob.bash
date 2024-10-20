#!/bin/bash
#SBATCH --job-name=pix2text_job
#SBATCH --partition=gpu              
#SBATCH --gres=gpu:8                
#SBATCH --nodes=2           #optional                       
#SBATCH --time=10:00:00             
#SBATCH --output=pix2text_output_%j.log 
#SBATCH --error=pix2text_error_%j.log   

# Load the necessary module for Python
# Required packages already installed in the environment using "pip install pix2text onnxruntime-gpu"
module load python/3.9.15
module load cpe-cuda/23.03


# Activate virtual environment
source /scratch/courses0101/username/myenv/bin/activate

# Run the Python script with a specific PDF chunk
python run_pix2text.py "output_${SLURM_ARRAY_TASK_ID}.pdf"
