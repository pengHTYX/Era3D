# export CONTAINER_PARAMS="--no-container-mount-home \
#  --container-remap-root --container-writable \
#  --container-mounts=/scratch/vgenfmod/limengfei:/scratch/vgenfmod/limengfei,/home/mliek:/home/mliek,/scratch/vgenfmod/lipeng:/scratch/vgenfmod/lipeng \
#  --container-image /scratch/vgenfmod/lipeng/wondersync/blender_render/render_container.sqsh \
#  --container-workdir=/scratch/vgenfmod/lipeng/wondersync/blender_render";

export CONTAINER_PARAMS="--no-container-mount-home \
 --container-remap-root --container-writable \
 --container-mounts=/scratch/vgenfmod/lipeng:/scratch/vgenfmod/lipeng\
 --container-image /scratch/vgenfmod/lipeng/wondersync/blender_render/render_container.sqsh \
 --container-workdir=/scratch/vgenfmod/lipeng/Era3D/blender"
