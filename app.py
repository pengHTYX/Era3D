import os
import torch
import fire
import gradio as gr
from PIL import Image
from functools import partial
import spaces 
import cv2
import time
import numpy as np
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor

import os
import torch

from PIL import Image
from typing import Dict, Optional,  List
from dataclasses import dataclass
from mvdiffusion.data.single_image_dataset import SingleImageDataset 
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from einops import rearrange
import numpy as np
import subprocess
from datetime import datetime
from icecream import ic
def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    return ndarr


def save_image_to_disk(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


weight_dtype = torch.float16

_TITLE = '''Era3D: High-Resolution Multiview Diffusion using Efficient Row-wise Attention'''
_DESCRIPTION = '''
<div>
Generate consistent high-resolution multi-view normals maps and color images.
</div>
<div>
The demo does not include the mesh reconstruction part, please visit <a href="https://github.com/pengHTYX/Era3D"><img src='https://img.shields.io/github/stars/pengHTYX/Era3D?style=social' style="display: inline-block; vertical-align: middle;"/></a> to get a textured mesh.
</div>
'''
_GPU_ID = 0


if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image


def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{_GPU_ID}")
    predictor = SamPredictor(sam)
    return predictor

@spaces.GPU
def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Background Removal" in chk_group
        rescale = "Rescale" in chk_group
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w] = image_arr[y : y + h, x : x + w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)

def load_era3d_pipeline(cfg):
    # Load scheduler, tokenizer and models.
    
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    )
    # sys.main_lock = threading.Lock()
    return pipeline


from mvdiffusion.data.single_image_dataset import SingleImageDataset


def prepare_data(single_image, crop_size, cfg):
    dataset = SingleImageDataset(root_dir='', num_views=6, img_wh=[512, 512], bg_color='white', 
        crop_size=crop_size, single_image=single_image, prompt_embeds_path=cfg.validation_dataset.prompt_embeds_path)
    return dataset[0]

scene = 'scene'
@spaces.GPU
def run_pipeline(pipeline, cfg, single_image, guidance_scale, steps, seed, crop_size, chk_group=None):
    pipeline.to(device=f'cuda:{_GPU_ID}')
    pipeline.unet.enable_xformers_memory_efficient_attention()
    
    global scene
    # pdb.set_trace()

    if chk_group is not None:
        write_image = "Write Results" in chk_group

    batch = prepare_data(single_image, crop_size, cfg)

    pipeline.set_progress_bar_config(disable=True)
    seed = int(seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)


    imgs_in = torch.stack([imgs_in]*2, dim=0)
    num_views = imgs_in.shape[1]
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")# (B*Nv, 3, H, W)
    
    normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'], batch['color_prompt_embeddings'] 
    prompt_embeddings = torch.stack([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
    prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
    
    
    imgs_in = imgs_in.to(device=f'cuda:{_GPU_ID}', dtype=weight_dtype)
    prompt_embeddings = prompt_embeddings.to(device=f'cuda:{_GPU_ID}', dtype=weight_dtype)
    
    out = pipeline(
        imgs_in, 
        None, 
        prompt_embeds=prompt_embeddings,
        generator=generator, 
        guidance_scale=guidance_scale, 
        output_type='pt', 
        num_images_per_prompt=1, 
        # return_elevation_focal=cfg.log_elevation_focal_length,
        **cfg.pipe_validation_kwargs
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]
    num_views = 6
    if write_image:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        cur_dir = os.path.join(cfg.save_dir, f"cropsize-{int(crop_size)}-cfg{guidance_scale:.1f}")
        
        scene = 'scene'+datetime.now().strftime('@%Y%m%d-%H%M%S')
        scene_dir = os.path.join(cur_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)

        for j in range(num_views):
            view = VIEWS[j]
            normal = normals_pred[j]
            color = images_pred[j]

            normal_filename = f"normals_{view}_masked.png"
            color_filename = f"color_{view}_masked.png"
            normal = save_image_to_disk(normal, os.path.join(scene_dir, normal_filename))
            color = save_image_to_disk(color, os.path.join(scene_dir, color_filename))


    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]
    
    out = images_pred + normals_pred
    return images_pred, normals_pred


def process_3d(mode, data_dir, guidance_scale, crop_size):
    dir = None
    global scene

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    subprocess.run(
        f'cd instant-nsr-pl && bash run.sh 0 {scene} exp_demo && cd ..',
        shell=True,
    )
    import glob

    obj_files = glob.glob(f'{cur_dir}/instant-nsr-pl/exp_demo/{scene}/*/save/*.obj', recursive=True)
    print(obj_files)
    if obj_files:
        dir = obj_files[0]
    return dir


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
    


def run_demo():
    from utils.misc import load_config
    from omegaconf import OmegaConf

    # parse YAML config to OmegaConf
    cfg = load_config("./configs/test_unclip-512-6view.yaml")
    # print(cfg)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    pipeline = load_era3d_pipeline(cfg)
    torch.set_grad_enabled(False)

    
    predictor = sam_init()


    custom_theme = gr.themes.Soft(primary_hue="blue").set(
        button_secondary_background_fill="*neutral_100", button_secondary_background_fill_hover="*neutral_200"
    )
    custom_css = '''#disp_image {
        text-align: center; /* Horizontally center the content */
    }'''
    

    with gr.Blocks(title=_TITLE, theme=custom_theme, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(type='pil', image_mode='RGBA', height=320, label='Input image')

            with gr.Column(scale=1):
                processed_image_highres = gr.Image(type='pil', image_mode='RGBA', visible=False)
               
                processed_image = gr.Image(
                    type='pil',
                    label="Processed Image",
                    interactive=False,
                    # height=320,
                    image_mode='RGBA',
                    elem_id="disp_image",
                    visible=True,
                )
            # with gr.Column(scale=1):
            #     ## add 3D Model
            #     obj_3d = gr.Model3D(
            #                         # clear_color=[0.0, 0.0, 0.0, 0.0], 
            #                         label="3D Model", height=320, 
            #                         # camera_position=[0,0,2.0]
            #                         )
                
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                example_folder = os.path.join(os.path.dirname(__file__), "./examples")
                example_fns = [os.path.join(example_folder, example) for example in os.listdir(example_folder)]
                gr.Examples(
                    examples=example_fns,
                    inputs=[input_image],
                    outputs=[input_image],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=30,
                )
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion('Advanced options', open=True):
                            input_processing = gr.CheckboxGroup(
                                ['Background Removal'],
                                label='Input Image Preprocessing',
                                value=['Background Removal'],
                                info='untick this, if masked image with alpha channel',
                            )
                    with gr.Column():
                        with gr.Accordion('Advanced options', open=False):
                            output_processing = gr.CheckboxGroup(
                                ['Write Results'], label='write the results in mv_res folder', value=['Write Results']
                            )
                    with gr.Row():
                        with gr.Column():
                            scale_slider = gr.Slider(1, 5, value=3, step=1, label='Classifier Free Guidance Scale')
                        with gr.Column():
                            steps_slider = gr.Slider(15, 100, value=40, step=1, label='Number of Diffusion Inference Steps')
                    with gr.Row():
                        with gr.Column():
                            seed = gr.Number(600, label='Seed', info='100 for digital portraits')
                        with gr.Column():
                            crop_size = gr.Number(420, label='Crop size', info='380 for digital portraits')

                        mode = gr.Textbox('train', visible=False)
                        data_dir = gr.Textbox('outputs', visible=False)
                    # with gr.Row():
                    #     method = gr.Radio(choices=['instant-nsr-pl', 'NeuS'], label='Method (Default: instant-nsr-pl)', value='instant-nsr-pl')
                run_btn = gr.Button('Generate Normals and Colors', variant='primary', interactive=True)
                # recon_btn = gr.Button('Reconstruct 3D model', variant='primary', interactive=True)
                # gr.Markdown("<span style='color:red'>First click Generate button, then click Reconstruct button. Reconstruction may cost several minutes.</span>")
        
        with gr.Row():
            view_gallery = gr.Gallery(label='Multiview Images')
            normal_gallery = gr.Gallery(label='Multiview Normals')
            
        print('Launching...')
        run_btn.click(
            fn=partial(preprocess, predictor), inputs=[input_image, input_processing], outputs=[processed_image_highres, processed_image], queue=True
        ).success(
            fn=partial(run_pipeline, pipeline, cfg),
            inputs=[processed_image_highres, scale_slider, steps_slider, seed, crop_size, output_processing],
            outputs=[view_gallery, normal_gallery],
        )
        # recon_btn.click(
        #     process_3d, inputs=[mode, data_dir, scale_slider, crop_size], outputs=[obj_3d]
        # )

        demo.queue().launch(share=True, max_threads=80)
        

if __name__ == '__main__':
    fire.Fire(run_demo)