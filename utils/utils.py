from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
def make_grid_(imgs, save_file, nrow=10, pad_value=1):
    if isinstance(imgs, list):
        if isinstance(imgs[0], Image.Image):
            imgs = [torch.from_numpy(np.array(img)/255.) for img in imgs]
        elif isinstance(imgs[0], np.ndarray):
            imgs = [torch.from_numpy(img/255.) for img in imgs]
        imgs = torch.stack(imgs, 0).permute(0, 3, 1, 2)
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)

    img_grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=pad_value)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype(np.uint8)
    img_grid = Image.fromarray(img_grid)
    img_grid.save(save_file) 
    
def draw_caption(img, text, pos, size=100, color=(128, 128, 128)):
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(size= size)
    font = ImageFont.load_default()
    font = font.font_variant(size=size)
    draw.text(pos, text, color, font=font)
    return img