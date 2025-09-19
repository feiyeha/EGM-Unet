import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random
import torch

# class DriveDataset(Dataset):
#     def __init__(self, root: str, transforms=None, txt_name: str = "train.txt"):
#         super(DriveDataset, self).__init__()
#         data_root = os.path.join(root, "TP-Dataset")
#         assert os.path.exists(data_root), f"path '{data_root}' does not exists."
#         image_dir = os.path.join(data_root, 'JPEGImages')
#         mask_dir = os.path.join(data_root, 'GroundTruth')

#         txt_path = os.path.join(data_root, "Index", txt_name)
#         assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
#         with open(os.path.join(txt_path), 'r+') as f:
#             file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
#         self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
#         self.mask = [os.path.join(mask_dir, x + ".png") for x in file_names]
#         assert (len(self.images)) == len(self.mask)
#         self.transforms = transforms
#         self.class_mapping = {
#             0: 0,    # 背景
#             1: 1,   # 车
#             2: 2,    # 人行道
#             3: 3    # 盲道
#         }
#     def __getitem__(self, idx):
#         img = Image.open(self.images[idx]).convert('RGB')
#         # 直接读取原始标签值（不要归一化）
#         target = Image.open(self.mask[idx]).convert('L')
#         target_array = np.array(target)  # 保持原始整数值
        
#         # 创建映射后的标签
#         mapped_target = np.zeros_like(target_array, dtype=np.uint8)
#         for orig_val, mapped_idx in self.class_mapping.items():
#             mapped_target[target_array == orig_val] = mapped_idx
        
#         # 转换为PIL图像（模式为'L'）
#         mask = Image.fromarray(mapped_target)
        
#         if self.transforms is not None:
#             img, mask = self.transforms(img, mask)
            
#         return img, mask


# #     def __getitem__(self, idx):
# #         img = Image.open(self.images[idx]).convert('RGB')
# #         target = Image.open(self.mask[idx]).convert('L')
# #         target = np.array(target) / 255
# #         mask = np.clip(target, a_min=0, a_max=255)
# #         mask = Image.fromarray(mask)

# #         if self.transforms is not None:
# #             img, mask = self.transforms(img, mask)

# #         return img, mask

#     def __len__(self):
#         return len(self.images)
        

#     @staticmethod
#     def collate_fn(batch):
#         images, targets = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         # batched_targets = cat_list(targets, fill_value=255)
#         batched_targets = cat_list(targets, fill_value=0)
#         return batched_imgs, batched_targets


# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs



# 只用盲道
class DriveDataset(Dataset):
    def __init__(self, root: str, transforms=None, txt_name: str = "train.txt"):
        super(DriveDataset, self).__init__()
        data_root = os.path.join(root, "TP-Dataset")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        image_dir = os.path.join(data_root, 'JPEGImages')
        mask_dir = os.path.join(data_root, 'GroundTruth')

        txt_path = os.path.join(data_root, "Index", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), 'r+') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.mask = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images)) == len(self.mask)
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.mask[idx]).convert('L')
        target = np.array(target) / 255
        mask = np.clip(target, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

