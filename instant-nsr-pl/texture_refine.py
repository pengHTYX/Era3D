from utils.func import load_obj,  save_obj,  make_sparse_camera, calc_vertex_normals, make_round_views, make_addition_views
from utils.render import nvdiffRenderer
from utils.video_utils import *
import  torch.optim as optim
from tqdm import tqdm
import cv2
from rembg import remove
# try:
#     from util.view import show
# except:
#     show = None
import numpy as np
import os
from PIL import Image
import kornia
import torch
import torch.nn as nn
import  torch.nn.functional as F 
import trimesh
from icecream import ic
#### ------------------- config----------------------
steps = 200
lr_clr = 2e-3
scale = 1
bg_color = np.array([1,1,1], dtype=np.float32) #to prevent automatic float64 typecasting by numpy, ironing out the ambiguous declaration
cam_path = './datasets/fixed_poses'
views = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
view_nums = len(views)
addition_angles = [135, 225]
num_additions = len(addition_angles)
res=1024
def evaluate(target_vertices, target_colors, target_faces, device, save_path=None, save_nrm=False):
    mv, proj = make_round_views(60, scale)
    renderer = nvdiffRenderer(mv, proj, [res, res], device=device)
    
    target_images = renderer.render(target_vertices,target_faces, colors=target_colors)
    target_images = target_images.cpu().numpy()
    target_images = target_images[..., :3] * target_images[..., 3:4]  + bg_color * (1 - target_images[..., 3:4])
    target_images = (target_images.clip(0, 1) * 255).astype(np.uint8)
    
    if save_nrm:
        target_normals = calc_vertex_normals(target_vertices, target_faces)
        target_normals[:, 2] *= -1
        target_normals = renderer.render(target_vertices, target_faces, normals=target_normals)
        target_normals = target_normals.cpu().numpy()
        target_normals = target_normals[..., :3] * target_normals[..., 3:4]  + bg_color * (1 - target_normals[..., 3:4])
        target_normals = (target_normals.clip(0, 1) * 255).astype(np.uint8)
        frames = [np.concatenate([img, nrm], 1) for img, nrm in zip(target_images, target_normals)]
    else:
        frames = [img for img in target_images]
    if save_path is not None:
        write_video(frames, fps=25, save_path=save_path)
    return frames

def prepare_renderer(device):
    mv, proj = make_sparse_camera(cam_path, scale, device=device) # 6
    mv_addition, _ = make_addition_views(addition_angles, scale, device=device) 
    mv = torch.cat([mv, mv_addition], 0)    
    renderer = nvdiffRenderer(mv, proj, [res,res], device=device)
    return renderer
    
def load_training_data(img_path, obj_path, case, device):
    target_vertices, init_colors, target_faces =  load_obj(obj_path, device=device)

    colors = []
    for view in views:
        color = Image.open(f'{img_path}/{case}/color_{view}_masked.png')
        color = color.convert('RGBA').resize((res, res), Image.BILINEAR)
        color = np.array(color).astype(np.float32) / 255.
        color_mask = color[..., 3:]  # alpha
        color = color[..., :3] * color_mask  + bg_color * (1 - color_mask)
        colors.append(color)
    colors = np.stack(colors, 0)
    target_colors = torch.from_numpy(colors).to(device)
    return target_vertices, target_faces, init_colors, target_colors


class ColorModel(nn.Module):
    def __init__(self, v, f, c, device):
        super().__init__()
        # colors = torch.ones_like(v).float().cuda() * 0.5
        self.colors = nn.Parameter(c, requires_grad=True)
        self.bg_color = torch.from_numpy(bg_color).float().to(device)
        self.renderer = prepare_renderer(device)
        self.v = v
        self.f = f
        
    def forward(self):
        rgba = self.renderer.render(self.v, self.f, colors=self.colors)
        mask = rgba[..., 3:]
        return rgba[..., :3] * mask + self.bg_color * (1 - mask)
        

def optim_clr(case, img_path, mesh_dir, save_dir, device):
    vert, face, init_colors, target_colors = load_training_data(img_path, f'{mesh_dir}/it3000-mc256.obj', case, device)
    ###----------------------- refine color-------------------------------------
    weight = torch.Tensor([1., .2, .7, 1., .7, .2, ] + [0.,]*num_additions).view(view_nums+num_additions,1,1,1).to(device)
    color_model = ColorModel(vert, face, init_colors, device)
   
    clr_opt = optim.Adam(color_model.parameters(), lr=lr_clr)
    clr_scheduler = optim.lr_scheduler.MultiStepLR(clr_opt, milestones=[300, 800], gamma=0.1)
    for i in tqdm(range(steps)):
        clr_opt.zero_grad() 
        clr = color_model()
        if i == 0:
            clr_addi_neus = clr[view_nums:].detach()
            target_colors = torch.cat([target_colors, clr_addi_neus], 0)
        # if i == 200:
        #     weight = torch.Tensor([1., .7, .7, 1., .7, .7, ] + [0.4]*num_additions).view(view_nums+num_additions,1,1,1).to(device)
        loss = ((clr-target_colors) * weight).abs().mean()
        loss.backward()
        clr_opt.step()
        clr_scheduler.step()

    save_obj(vert, face, f'{save_dir}/refine_{case}.obj', color_model.colors.detach())
    return evaluate(vert, color_model.colors.detach(), face, device=device, save_nrm=False, save_path=f'{save_dir}/refine_{case}.mp4')

def crop_input(image_input):
    def add_margin(pil_img, color=0, size=256):
        width, height = pil_img.size
        result = Image.new(pil_img.mode, (size, size), color)
        result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
        return result

    crop_size = 400
    image_size = 512  
    alpha_np = np.asarray(image_input)[:, :, 3]
    coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
    min_x, min_y = np.min(coords, 0)
    max_x, max_y = np.max(coords, 0)
    ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
    h, w = ref_img_.height, ref_img_.width
    scale = crop_size / max(h, w)
    h_, w_ = int(scale * h), int(scale * w)
    ref_img_ = ref_img_.resize((w_, h_))
    image_input = add_margin(ref_img_, size=image_size)
    return image_input

def texture_refine(case, input_dir, img_path, mesh_dir, save_dir, device):
    color_frames = optim_clr(case, img_path, mesh_dir, save_dir, device)
    color_clip = ImageSequenceClip(color_frames, fps=25)
    try:
        img = Image.open(f'{input_dir}/{case}.png') 
    except:
        img = Image.open(f'{input_dir}/{case}.webp') 
    img = crop_input(img)
    img.save(f'{save_dir}/{case}.png')
    img = img.resize((res, res), Image.BILINEAR)
    img = np.array(img) / 255.
    if img.shape[-1] == 4:
        img = img[..., :3] + bg_color * (1 - img[..., 3:])
        img = (img * 255).astype(np.uint8)
    vclip = concat_img_video(color_clip, img) 
    
    normal_clip =  load_video(f'{mesh_dir}/it3000-test.mp4')
    vclip = concat_video_clips([vclip, normal_clip])
    write_video(vclip, fps=25, save_path=f'{save_dir}/refine_{case}.mp4')
    
if __name__ == '__main__':
    texture_refine(
                    'A_bulldog_with_a_black_pirate_hat_rgba', 
                    '../examples',
                    '../mv_res',
                    'recon/A_bulldog_with_a_black_pirate_hat_rgba/@20240514-221419/save', 
                    'recon', 
                    torch.device('cuda:0'))



