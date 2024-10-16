python launch.py \
    --config configs/neuralangelo-ortho-wmask.yaml \
    --gpu $1 \
    --train dataset.root_dir=../mv_res dataset.scene=$2 \
    --exp_dir $3 
