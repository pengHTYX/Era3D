from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.editor import VideoFileClip, clips_array
import os
import argparse
from PIL import Image
import numpy as np

bg_color = np.array([1,1,1])

def concat_imgs(imgs: list[Image.Image], size=(512, 512), mode='horizontal'):
    num = len(imgs)
    if mode == 'horizontal':
        w_ = size[0] * num
        h_ = size[1]
        new_img = Image.new('RGB', (w_, h_))
        for i, img in enumerate(imgs):
            if img.mode == 'RGBA':
                img = np.array(img) / 255.
                img = img[..., :3] + bg_color * (1 - img[..., 3:])
                img = (img * 255).astype(np.uint8) 
                img = Image.fromarray(img)
                img = img.resize(size)
            new_img.paste(img, (i*size[0], 0))
    elif mode == 'vertical':
        w_ = imgs[0].width
        h_ = size[1] * num
        new_img = Image.new('RGB', (w_, h_))
        for i, img in enumerate(imgs):
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            new_img.paste(img, (0, i*size[1]))
    return new_img

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
    
    img = np.array(image_input) / 255.
    if img.shape[-1] == 4:
        img = img[..., :3] + bg_color * (1 - img[..., 3:])
        img = (img * 255).astype(np.uint8)
    image_input = Image.fromarray(img)
    return image_input


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--img_path', type=str, default='wonder3d_recon_res')
    args.add_argument('--mv_path', type=str, default='wonder3d_recon_res')
    args.add_argument('--out_dir', type=str, default='wonder3d_recon_res')
    args = args.parse_args()
    
    cases = os.listdir(args.img_path)
    
    os.makedirs(args.out_dir, exist_ok=True)
    views = ['color_front_masked.png', 'color_back_masked.png', 'color_left_masked.png',
             'normals_front_masked.png', 'normals_back_masked.png', 'normals_left_masked.png',]
    concat_list = []
    for case in cases:
        img_list = [crop_input(Image.open(os.path.join(args.img_path, case)))]
        case_name = case.split('.')[0]
        for v in views:
            img_list.append(Image.open(os.path.join(args.mv_path, case_name, v))) 
        img = concat_imgs(img_list, mode='horizontal')
        img.save(os.path.join(args.out_dir, f'debug.png'))
        concat_list.append(img)
    concat_ = concat_imgs(concat_list, mode='vertical')
    concat_.save(os.path.join(args.out_dir, f'more_result.png'))