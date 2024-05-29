python launch.py --config configs/neuralangelo-ortho-wmask.yaml --gpu 0 --test dataset.root_dir='mv_res/steplr/webcase'\
    dataset.scene=backpack \
    --resume 'exp/moria-000-135/@20240311-200730/ckpt/epoch=0-step=3000.ckpt' \
    --exp_dir exp_demo
