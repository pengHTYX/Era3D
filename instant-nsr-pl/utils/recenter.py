from PIL import Image
import numpy as np

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


image_path = 'evaluate/webcase/dslr.png'
image_input = Image.open(image_path)    
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

image_input.save('./dslr.png')