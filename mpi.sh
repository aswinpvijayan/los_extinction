#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=mpipy
#SBATCH --array=0-38%19
#SBATCH -t 0-4:00
#SBATCH --ntasks 8
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/slurm.%N.%j.out.txt
#SBATCH -e logs/slurm.%N.%j.err.txt

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 python/3.6.5


mpiexec -n 8 python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID '010_z005p000'
mpiexec -n 8 python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID '009_z006p000'
mpiexec -n 8 python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID '008_z007p000'
mpiexec -n 8 python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID '007_z008p000'
mpiexec -n 8 python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID '006_z009p000'
mpiexec -n 8 python3 get_GEAGLE.py $SLURM_ARRAY_TASK_ID '005_z010p000'

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
