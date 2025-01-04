#!/bin/bash
#SBATCH -J dolma_dataset_job
#SBARCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -A BUTTERY-SL2-CPU
#SBATCH -p icelake-himem
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

cd pico 
source $(poetry env info --path)/bin/activate
cd ..

scontrol update job $SLURM_JOB_ID name=dolma_dataset_job_idx$1

echo "Starting shard $1"
python create_dolma_dataset.py --idx $1 --num_workers ${2:-14}
