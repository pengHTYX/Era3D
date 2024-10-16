import argparse
import os
from typing import Dict, Optional,  List
from omegaconf import OmegaConf
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from accelerate.utils import  set_seed
from tqdm.auto import tqdm
from mvdiffusion.data.single_image_dataset import SingleImageDataset
from einops import rearrange, repeat
from rembg import remove
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline

weight_dtype = torch.float16

def tensor_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:Optional[str]
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    # save_single_views: bool
    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation
    regress_elevation: bool
    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool
    
    regress_elevation: bool
    regress_focal_length: bool
    


def convert_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def save_image(tensor, fp):
    ndarr = convert_to_numpy(tensor)
    # pdb.set_trace()
    save_image_numpy(ndarr, fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

def log_validation_joint(dataloader, pipeline, cfg: TestConfig,  save_dir):

    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.unet.device).manual_seed(cfg.seed)
    
    images_cond, pred_cat = [], defaultdict(list)
    for _, batch in tqdm(enumerate(dataloader)):
        images_cond.append(batch['imgs_in'][:, 0]) 
        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")# (B*Nv, 3, H, W)

        normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'], batch['color_prompt_embeddings'] 
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")

        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                unet_out = pipeline(
                    imgs_in, None, prompt_embeds=prompt_embeddings,
                    generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, 
                    **cfg.pipe_validation_kwargs
                )
                
                out = unet_out.images
                bsz = out.shape[0] // 2

                normals_pred = out[:bsz]
                images_pred = out[bsz:] 
                # print(normals_pred.shape, images_pred.shape)
                pred_cat[f"cfg{guidance_scale:.1f}"].append(torch.cat([normals_pred, images_pred], dim=-1)) # b, 3, h, w
                # cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}-seed{cfg.seed}")
                cur_dir = save_dir 
                os.makedirs(cur_dir, exist_ok=True)
                if cfg.save_mode == 'concat': ## save concatenated color and normal---------------------
                    for i in range(bsz//num_views):
                        scene =  batch['filename'][i].split('.')[0]

                        img_in_ = images_cond[-1][i].to(out.device)
                        vis_ = [img_in_]
                        for j in range(num_views):
                            view = VIEWS[j]
                            idx = i*num_views + j
                            normal = normals_pred[idx]
                            color = images_pred[idx]
                           
                            vis_.append(color)
                            vis_.append(normal)

                        out_filename = f"{cur_dir}/{scene}.png"
                        vis_ = torch.stack(vis_, dim=0)
                        vis_ = make_grid(vis_, nrow=len(vis_), padding=0, value_range=(0, 1))
                        save_image(vis_, out_filename)
                elif cfg.save_mode == 'rgb':
                    for i in range(bsz//num_views):
                        scene =  batch['filename'][i].split('.')[0]
                        scene_dir = os.path.join(cur_dir, scene)
                        os.makedirs(scene_dir, exist_ok=True)

                        img_in_ = images_cond[-1][i].to(out.device)
                        vis_ = [img_in_]
                        for j in range(num_views):
                            view = VIEWS[j]
                            idx = i*num_views + j
                            normal = normals_pred[idx]
                            color = images_pred[idx]
                            vis_.append(color)
                            vis_.append(normal)

                            ## save color and normal---------------------
                            normal_filename = f"normals_{view}_masked.png"
                            rgb_filename = f"color_{view}_masked.png"
                            save_image(normal, os.path.join(scene_dir, normal_filename))
                            save_image(color, os.path.join(scene_dir, rgb_filename))
                elif cfg.save_mode == 'rgba':

                    for i in range(bsz//num_views):
                        scene =  batch['filename'][i].split('.')[0]
                        scene_dir = os.path.join(cur_dir, scene)
                        os.makedirs(scene_dir, exist_ok=True)

                        img_in_ = images_cond[-1][i].to(out.device)
                        vis_ = [img_in_]
                        for j in range(num_views):
                            view = VIEWS[j]
                            idx = i*num_views + j
                            normal = normals_pred[idx]
                            color = images_pred[idx]
                            vis_.append(color)
                            vis_.append(normal)
                            
                            normal = convert_to_numpy(normal)
                            color = convert_to_numpy(color)
                            rm_normal = remove(normal)
                            rm_color = remove(color)
                            normal_filename = f"normals_{view}_masked.png"
                            rgb_filename = f"color_{view}_masked.png"
                            save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
                            save_image_numpy(rm_color, os.path.join(scene_dir, rgb_filename))
    torch.cuda.empty_cache()    

def load_era3d_pipeline(cfg):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline

def main(
    cfg: TestConfig
):
    if cfg.seed is not None:
        set_seed(cfg.seed)
    pipeline = load_era3d_pipeline(cfg)
    if torch.cuda.is_available():
        pipeline.to('cuda:0')

    # Get the  dataset
    validation_dataset = SingleImageDataset(
        **cfg.validation_dataset
    )
    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )
    os.makedirs(cfg.save_dir, exist_ok=True)

    log_validation_joint(validation_dataloader, pipeline, cfg, cfg.save_dir)
   
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()

    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    print(cfg)
    schema = OmegaConf.structured(TestConfig)
    # cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    if cfg.num_views == 6:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    elif cfg.num_views == 4:
        VIEWS = ['front', 'right', 'back', 'left']
    elif cfg.num_views == 8:
        VIEWS = ['front', 'front_right', 'right', 'back_right', 'back', 'back_left', 'left', 'front_left']
    main(cfg)
