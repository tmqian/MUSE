#!/bin/bash

#SBATCH --job-name=g4a

##SBATCH --partition=general # partition (unrestricted options are dawson, ellis and kruskal)

#SBATCH -A pppl
#SBATCH -n 32 # number of cores
#SBATCH -N 2 # number of nodes
##SBATCH --mem 32G # memory to be used per node
##SBATCH -C piledriver

#SBATCH -t 0-8:00 # time (D-HH:MM)

#SBATCH -o slurm.%N.%j.out # STDOUT

#SBATCH -e slurm.%N.%j.err # STDERR

#SBATCH --mail-type=FAIL # notifications for job done & fail

#SBATCH --mail-user=tqian@pppl.gov # send-to address


#module load mod_focus
module use ~caoxiang/Modules
module load focus/dipole
#module load openmpi

# 'Normal' MPI run command
#

#srun ~/CODE/FOCUS/bin/xfocus 16pga-PA1-1-0p0005-0p0005.input > log.16pga-PA1-1-0p0005-0p0005
