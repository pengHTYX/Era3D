#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH -p project
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/output_render.txt
#SBATCH --error=slurm_logs/error_render.txt


export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="^docker0,lo,bond0"
export WORK_DIR=/scratch/suptest/3d/codes/wondersync/blender_render
export PYTHONPATH=$WORK_DIR
export ENROOT_RESTRICT_DEV=n

export CONTAINER_PARAMS="--no-container-mount-home \
 --container-remap-root --container-writable \
 --container-mounts=/scratch/vgenfmod/lipeng:/scratch/vgenfmod/lipeng \
 --container-image /scratch/suptest/3d/render_container.sqsh \
 --container-workdir=/scratch/suptest/3d/codes";
echo "Running with container params: $CONTAINER_PARAMS"
srun python test.py


source /scratch/suptest/3d/codes/container.sh && ENROOT_RESTRICT_DEV=n srun -p project --gpus-per-task=1 --ntasks-per-node=1 --nodes=1 --time=24:00:00 --cpus-per-task=16  python test.py
