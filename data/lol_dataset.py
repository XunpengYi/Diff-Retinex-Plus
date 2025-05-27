import cv2
import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import glob
import numpy as np

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from scripts.utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding

@DATASET_REGISTRY.register()
class LoLDataset(data.Dataset):
    """Example dataset.
    1. Read GT image
    2. Generate LQ (Low Quality) image with cv2 bicubic downsampling and JPEG compression
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(LoLDataset, self).__init__()
        self.opt = opt
        self.gt_root = opt['gt_root']
        self.input_root = opt['input_root']

        self.gt_paths = glob.glob(os.path.join(self.gt_root, '*.png')) + glob.glob(os.path.join(self.gt_root, '*.jpg'))

        self.mean = self.opt['mean']
        self.std = self.opt['std']

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        gt_name = os.path.split(gt_path)[-1]

        input_path = os.path.join(self.input_root, gt_name)
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255.

        if self.opt.get('LL_mixup_aug', False):
            if np.random.uniform() < 0.4:
                LL_mixup_aug_range = self.opt.get('LL_mixup_aug_range', [0.5, 1.])
                input_img = input_img * np.random.uniform(*LL_mixup_aug_range) + gt_img * (1 - np.random.uniform(*LL_mixup_aug_range))
            else:
                input_img = input_img

        if self.opt.get('bright_aug', False):
            bright_aug_range = self.opt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)

        if self.opt.get('concat_with_hiseq', False):
            hiseql = cv2.cvtColor(hiseq_color_cv2_img(cv2.imread(input_path)), cv2.COLOR_BGR2RGB) / 255.
            if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p', 0.5):
                input_img = np.concatenate([hiseql, input_img], axis=2)
            else:
                input_img = np.concatenate([input_img, hiseql], axis=2)
            if self.opt.get('random_drop', False):
                if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                    random_drop_val = self.opt.get('random_drop_val', 0)
                    if np.random.uniform() < 0.5:
                        input_img[:, :, :3] = random_drop_val
                    else:
                        input_img[:, :, 3:] = random_drop_val
            if self.opt.get('random_drop_hiseq', False):
                if np.random.uniform() < 0.5:
                    input_img[:, :, 3:] = 0

        if self.opt.get('use_flip', False) and np.random.uniform() < 0.5:
            gt_img = cv2.flip(gt_img, 1, gt_img)
            input_img = cv2.flip(input_img, 1, input_img)

        if self.opt['input_mode'] == 'crop':
            crop_size = self.opt['crop_size']
            H, W, _ = input_img.shape
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
            h = np.random.randint(0, H - crop_size + 1)
            w = np.random.randint(0, W - crop_size + 1)
            gt_img = gt_img[h: h + crop_size, w: w + crop_size, :]
            input_img = input_img[h: h + crop_size, w: w + crop_size, :]
        if self.opt['input_mode'] == 'pad':
            divide = self.opt['divide']
            gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
            input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
            gt_img_pt = torch.unsqueeze(gt_img_pt, 0)
            input_img_pt = torch.unsqueeze(input_img_pt, 0)
            gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
            input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
            gt_img_pt = gt_img_pt[0, ...]
            input_img_pt = input_img_pt[0, ...]
            gt_img = gt_img_pt.numpy().transpose((1, 2, 0))
            input_img = input_img_pt.numpy().transpose((1, 2, 0))

        gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))

        input_img_pt = input_img_pt.float()
        gt_img_pt = gt_img_pt.float()

        return_dict = {'lq': input_img_pt, 'gt': gt_img_pt, 'lq_path': input_path, 'gt_path': gt_path}
        if self.opt['input_mode'] == 'pad':
            return_dict["pad_left"] = pad_left
            return_dict["pad_right"] = pad_right
            return_dict["pad_top"] = pad_top
            return_dict["pad_bottom"] = pad_bottom

        return return_dict

    def __len__(self):
        return len(self.gt_paths)
