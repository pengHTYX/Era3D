source ./container.sh &&  srun -p project -t 48:00:00 --nodes=1 --gpus-per-task=1 --ntasks-per-node=1  $CONTAINER_PARAMS --cpus-per-task=16 --pty bash


