#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=py
#SBATCH -t 0-4:00
#SBATCH --ntasks 16
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/slurm.%N.%j.out.txt 
#SBATCH -e logs/slurm.%N.%j.err.txt

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 python/3.6.5 

mpiexec -n 16 python3 get_eagle.py 

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
