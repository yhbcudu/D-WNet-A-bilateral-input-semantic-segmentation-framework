import os
from torch.utils.data import Dataset
import rasterio
import numpy as np
class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.names = os.listdir(os.path.join(path,'train4band'))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image_path = os.path.join(self.path, 'train4band', name)
        label_path = os.path.join(self.path, 'trainlable', name)
        # 使用Rasterio库打开图像
        with rasterio.open(image_path) as src:
            image = src.read()
        with rasterio.open(label_path) as src:
            label = src.read()
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label



print("Date Succeed!")