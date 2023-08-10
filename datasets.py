import glob
import os
import random
import torch

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


def tophat(img, fsize=20, return_img=True):
    if not isinstance(img, np.ndarray):
        img = np.array(img).astype(np.float32)
    filterSize = (fsize, fsize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    wth = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    bth = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    dst = (img + wth - bth).clip(0, 255).astype(np.uint8)
    if return_img:
        return T.ToPILImage()(dst)
    else:
        return dst


def random_rotate(img, p=1, angles=[0, 90, 180, 270]):
    if random.random() < p:
        angle = random.choice(angles)
        return T.functional_pil.rotate(img, angle)
    else:
        return img


def random_gamma(img, scale=(0.1, 2.2), p=1):
    if random.random() < p:
        gamma = random.uniform(scale[0], scale[1])
        return T.functional_pil.adjust_gamma(img, gamma)
    else:
        return img

def random_tophat(img, p=1):
    if random.random() < p:
        return tophat(img, random.randint(50, 100))
    else:
        return img


class ImageDataset(Dataset):
    def __init__(self, root, size=256, unaligned=False, mode='train', transform=None, return_img_name=False):
        if mode == 'train':
            self.transform = T.Compose([
                T.Lambda(lambda img:random_tophat(img, p=0.5)),  # random_tophat
                T.RandomResizedCrop(size, scale=(0.6, 1), interpolation=Image.BICUBIC),
                #  T.RandomEqualize(0.2),
                #  T.Resize(size),
                #  T.RandomAutocontrast(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                #  T.RandomVerticalFlip(p=0.5),
                T.Lambda(lambda x:random_rotate(x, p=1, angles=[0, 180])),  # 旋转
                T.ColorJitter(0.3, 0.3),
                T.Lambda(lambda x:random_gamma(x, (0.2, 2), p=1)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = T.Compose([
                #  T.Lambda(lambda img: tophat(img, 50)),  # TopHat增强
                T.Resize(size),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        if transform:
            self.transform = transform

        self.unaligned = unaligned

        self.return_img_name = return_img_name
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('L'))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))

        if self.return_img_name:
            A_name = Path(self.files_A[index % len(self.files_A)]).stem 
            B_name = Path(self.files_B[index % len(self.files_B)]).stem 
            return {'A': item_A, 'B': item_B, 'A_name': A_name, 'B_name': B_name}
        else:
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class ImageDataset2(Dataset):
    # 用于监督学习微调图像分割的数据集加载器，对图像和掩膜进行相同的变换。qq
    def __init__(self, root, size=256, unaligned=False, mode='train', transform=None, return_img_name=False):
        if mode == 'train':
            self.transform = T.Compose([
                T.Lambda(lambda img:random_tophat(img, p=0.5)), # random_tophat
                T.RandomResizedCrop(size, scale=(0.6, 1), interpolation=Image.BICUBIC),
                #  T.RandomEqualize(0.2),
                #  T.Resize(size),
                #  T.RandomAutocontrast(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                #  T.RandomVerticalFlip(p=0.5),
                T.Lambda(lambda x:random_rotate(x, p=1, angles=[0, 180])),  # 旋转
                T.ColorJitter(0.3, 0.3),
                T.Lambda(lambda x:random_gamma(x, (0.2, 2), p=1)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = T.Compose([
                #  T.Lambda(lambda img: tophat(img, 50)),  # TopHat增强
                T.Resize(size),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        if transform:
            self.transform = transform

        self.unaligned = unaligned

        self.return_img_name = return_img_name
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')
        item_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        # Merge A and B into a multi-channel image
        item_AB = Image.merge("RGB", (item_A, item_B, Image.new("L", item_A.size)))

        # Apply the transform to the merged image
        item_AB = self.transform(item_AB)
        
        # Split them back into separate tensors
        item_A = item_AB[0,:,:]
        item_B = item_AB[1,:,:]
        item_B = torch.sign(item_B)

        if self.return_img_name:
            A_name = Path(self.files_A[index % len(self.files_A)]).stem 
            B_name = Path(self.files_B[index % len(self.files_B)]).stem 
            return {'A': item_A, 'B': item_B, 'A_name': A_name, 'B_name': B_name}
        else:
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))



if __name__ == "__main__":
    import torch
    ts = T.Compose([
        T.RandomResizedCrop(512, scale=(0.6, 1), interpolation=Image.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.3, 0.3),
    ])

    x = torch.rand(2, 1, 512, 512)
    x = torch.autograd.Variable(x, requires_grad=True)
    y = ts(x)
    y[0, 0, 0, 0].backward()
    print(y)
