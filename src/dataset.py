import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mode='trainval', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.data = []

        if mode == 'trainval':
            sub_path = os.path.join(root_dir, 'VOC2012_train_val', 'VOC2012_train_val')
            self.image_dir = os.path.join(sub_path, 'JPEGImages')
            self.mask_dir = os.path.join(sub_path, 'SegmentationClass')
            self.filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(self.mask_dir)])
            
            for idx in range(len(self.filenames)):
                self.data.append(self.__read_data__(idx))
        else:
            sub_path = os.path.join(root_dir, 'VOC2012_test', 'VOC2012_test')
            self.image_dir = os.path.join(sub_path, 'JPEGImages')
            self.filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(self.image_dir)])

    def __read_data__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, f'{img_name}.jpg')).convert("RGB").resize((300, 300), Image.BILINEAR)
        mask = Image.open(os.path.join(self.mask_dir, f'{img_name}.png')).resize((300, 300), Image.NEAREST)
        return T.ToTensor()(image), torch.from_numpy(np.array(mask)).long()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.mode == 'trainval':
            image, mask = self.data[idx]
            if self.transform:
                image = self.transform(image)
            return image, mask
        else:
            img_name = self.filenames[idx]
            image = Image.open(os.path.join(self.image_dir, f'{img_name}.jpg')).convert("RGB")
            w,h = image.size
            image = image.resize((300, 300), Image.BILINEAR)
            if self.transform:
                image = self.transform(image)
            else:
                image = T.ToTensor()(image)
            return img_name, image, (w,h)