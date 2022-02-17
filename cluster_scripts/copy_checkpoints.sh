#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=2

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=01:00:00


# =====================
# Logging information
# =====================


# ===================
# Environment setup

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}
repo_home=/home/${USER}/git/laughter-detection/
# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/icsi/checkpoints
dest_path=${repo_home}/checkpoints/icsi/
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}
