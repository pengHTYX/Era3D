from typing import Dict
import numpy as np

import torch
from torch.utils.data import Dataset
import json
from PIL import Image

from typing import  Tuple, Optional, Any
import random

import json
import os


from PIL import Image
from normal_utils import worldNormal2camNormal, plot_grid_images, img2normal, norm_normalize, deg2rad

import pdb
from icecream import ic
def shift_list(lst, n):
    length = len(lst)
    n = n % length  # Ensure n is within the range of the list length
    return lst[-n:] + lst[:-n]


class ObjaverseDataset(Dataset):
    def __init__(self,
        root_dir: str,
        azi_interval: float,
        random_views: int,
        predict_relative_views: list,
        bg_color: Any,
        object_list: str,
        prompt_embeds_path: str,
        img_wh: Tuple[int, int],
        validation: bool = False,
        num_validation_samples: int = 64,
        num_samples: Optional[int] = None,
        invalid_list: Optional[str] = None,
        # augment_data: bool = False,
        side_views_rate: float = 0.,
        exten: str = '.png',
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.fixed_views = int(360 // azi_interval)
        self.bg_color = bg_color
        self.validation = validation
        self.num_samples = num_samples
        self.invalid_list = invalid_list
        self.img_wh = img_wh
        self.random_views = random_views
        self.total_views = int(self.fixed_views * (self.random_views + 1))
        self.predict_relative_views = predict_relative_views
        self.pred_view_nums = len(self.predict_relative_views)
        self.exten = exten
        self.side_views_rate = side_views_rate

        # ic(self.augment_data)
        ic(self.total_views)
        ic(self.fixed_views)
        ic(self.predict_relative_views)
        
        self.objects = []
        if object_list is not None:
            for dataset_list in object_list:
                with open(dataset_list, 'r') as f:
                #     objects = f.readlines()
                #     objects = [o.strip() for o in objects]
                    objects = json.load(f)
                self.objects.extend(objects)
        else:
            self.objects = os.listdir(self.root_dir)

        # load fixed camera poses
        self.trans_cv2gl_mat = np.linalg.inv(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        self.fix_cam_poses = []
        camera_path = os.path.join(self.root_dir, self.objects[0], 'camera')
        for vid in range(0, self.total_views, self.random_views+1):
            cam_info = np.load(f'{camera_path}/{vid:03d}.npy', allow_pickle=True).item()
            assert cam_info['camera'] == 'ortho', 'Only support predict ortho camera !!!'
            self.fix_cam_poses.append(cam_info['extrinsic'])
        random.shuffle(self.objects)
        
        invalid_objects = []
        if self.invalid_list is not None:
            for invalid_list in self.invalid_list:
                if invalid_list[-4:] == '.txt':
                    with open(invalid_list, 'r') as f:
                        sub_invalid = f.readlines()
                        invalid_objects.extend([o.strip() for o in sub_invalid]) 
                else:
                    with open(invalid_list) as f:
                        invalid_objects.extend(json.load(f))
        self.invalid_objects = invalid_objects
        ic(len(self.invalid_objects))
        
          
        self.all_objects = set(self.objects) - (set(self.invalid_objects) & set(self.objects))
        self.all_objects = list(self.all_objects)
        
        self.validation = validation
        if not validation:
            self.all_objects = self.all_objects[:-num_validation_samples]
            # print('Warning: you are fitting in small-scale dataset')
            # self.all_objects = self.all_objects
        else:
            self.all_objects = self.all_objects[-num_validation_samples:]
            
        if num_samples is not None:
            self.all_objects = self.all_objects[:num_samples]
        ic(len(self.all_objects))
        print("loading ", len(self.all_objects), " objects in the dataset")

        self.normal_prompt_embedding = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
        self.color_prompt_embedding = torch.load(f'{prompt_embeds_path}/clr_embeds.pt')
        
        self.backup_data = self.__getitem_norm__(0, '8609cf7e67bf413487a7d94c73aeaa3e') 
    
    def trans_cv2gl(self, rt):
        r, t = rt[:3, :3], rt[:3, -1]
        r = np.matmul(self.trans_cv2gl_mat, r)   
        t = np.matmul(self.trans_cv2gl_mat, t)
        return np.concatenate([r, t[:, None]], axis=-1)

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
        
            
    def load_image(self, img_path, bg_color, alpha=None, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        rgba = np.array(Image.open(img_path).resize(self.img_wh))
        rgba = rgba.astype(np.float32) / 255. # [0, 1]
        
        img = rgba[..., :3]
        if alpha is None:
            assert rgba.shape[-1] == 4 
            alpha = rgba[..., 3:4]
        assert alpha.sum() > 1e-8, 'w/o foreground'
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
    
    
    def load_normal(self, img_path, bg_color, alpha,  RT_w2c_cond=None, return_type='np'):
        normal_np = np.array(Image.open(img_path).resize(self.img_wh))[:, :, :3]
        assert np.var(normal_np) > 1e-8, 'pure normal'
        normal_cv = img2normal(normal_np)
        
        normal_relative_cv = worldNormal2camNormal(RT_w2c_cond[:3, :3], normal_cv)
        normal_relative_cv = norm_normalize(normal_relative_cv)
        # normal_relative_gl = normal_relative_cv[..., [ 0, 2, 1]]
        # normal_relative_gl[..., 2] = -normal_relative_gl[..., 2]
        normal_relative_gl = normal_relative_cv
        normal_relative_gl[..., 1:] = -normal_relative_gl[..., 1:]

        img = (normal_relative_cv*0.5 + 0.5).astype(np.float32)  # [0, 1]
        
        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img

    def __len__(self):
        return len(self.all_objects)
        
    def __getitem_norm__(self, index, debug_object=None):
        # get the bg color
        bg_color = self.get_bg_color()
        if debug_object is not  None:
            object_name =  debug_object
        else:
            object_name = self.all_objects[index % len(self.all_objects)]
            
        if self.validation:
            cond_ele0_idx = 12
        else:
            rand = random.random()
            if rand < self.side_views_rate: # 0.1
                cond_ele0_idx =  random.sample([8, 0], 1)[0]
            elif rand < 3 * self.side_views_rate: # 0.3
                cond_ele0_idx =  random.sample([10, 14], 1)[0]
            else:
                cond_ele0_idx = 12  #  front view
        cond_random_idx = random.sample(range(self.random_views+1), 1)[0]
        
        # condition info
        cond_ele0_vid = cond_ele0_idx * (self.random_views + 1)
        cond_vid = cond_ele0_vid + cond_random_idx   
        cond_ele0_w2c = self.fix_cam_poses[cond_ele0_idx]
        cond_info = np.load(f'{self.root_dir}/{object_name}/camera/{cond_vid:03d}.npy', allow_pickle=True).item()
        cond_type = cond_info['camera']
        focal_len = cond_info['focal']

        cond_eles = np.array([deg2rad(cond_info['elevation'])])
        
        img_tensors_in = [
            self.load_image(f"{self.root_dir}/{object_name}/image/{cond_vid:03d}{self.exten}", bg_color, return_type='pt')[0].permute(2, 0, 1)
        ] * self.pred_view_nums

        # output info
        pred_vids = [(cond_ele0_vid + i * (self.random_views+1)) % self.total_views  for i in self.predict_relative_views]
        # pred_w2cs = [self.fix_cam_poses[(cond_ele0_idx + i) % self.fixed_views] for i in self.predict_relative_views]
        img_tensors_out = []
        normal_tensors_out = []
        for vid in pred_vids:
            img_tensor, alpha_ = self.load_image(f"{self.root_dir}/{object_name}/image/{vid:03d}{self.exten}", bg_color, return_type='pt')

                                  
            img_tensor = img_tensor.permute(2, 0, 1) # (3, H, W)
            img_tensors_out.append(img_tensor)
            

            normal_tensor = self.load_normal(f"{self.root_dir}/{object_name}/normal/{vid:03d}{self.exten}", bg_color, alpha_.numpy(), RT_w2c_cond=cond_ele0_w2c[:3, :], return_type="pt").permute(2, 0, 1)
            normal_tensors_out.append(normal_tensor)


        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float() # (Nv, 3, H, W)

        elevations_cond = torch.as_tensor(cond_eles).float()
        if cond_type == 'ortho':
            focal_embed = torch.tensor([0.])
        else:
            focal_embed = torch.tensor([24./focal_len])
      
        
        
        return {            
                'elevations_cond': elevations_cond,
                'focal_cond': focal_embed,
                'id': object_name,
                'vid':cond_vid,
                'imgs_in': img_tensors_in,
                'imgs_out': img_tensors_out,
                'normals_out': normal_tensors_out,
                'normal_prompt_embeddings': self.normal_prompt_embedding,
                'color_prompt_embeddings': self.color_prompt_embedding
            }
       
            
  
    def __getitem__(self, index):
        try:
            return self.__getitem_norm__(index)
        except:
            print("load error ", self.all_objects[index%len(self.all_objects)] )
            return self.backup_data

        

    
        
       
