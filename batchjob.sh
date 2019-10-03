#!/bin/bash
#SBATCH --ntasks 1
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=python
#SBATCH --array=0-38%19 
#SBATCH -t 0-1:30
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J
 

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 python/3.6.5

python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
