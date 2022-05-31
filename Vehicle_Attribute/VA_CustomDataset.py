import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv

color_labels = ["white", "gray", "yellow", "red", "green", "blue", "black"]
type_labels = ["car", "bus", "truck", "van"]


class VehicleAttrsDataset(Dataset):
    def __init__(self, root_dir):
        # transform老样子
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((72, 72)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_files = os.listdir(root_dir)

        # mac系统的一个大坑，需要注意
        # 排除.DS_store文件
        for item in img_files:
            if item.startswith('.') and os.path.isfile(os.path.join(root_dir, item)):
                img_files.remove(item)

        nums_ = len(img_files)
        # vehicle type:  [car:0, bus:1, truck:2, van:3]
        # vehicle color:[white:0, gray:1, yellow:2, red:3, green:4, blue:5, black:6]
        self.vehicle_types = []
        self.vehicle_colors = []
        self.images = []
        index = 0
        for file_name in img_files:
            color_type_group = file_name.split("_")
            color_ = color_labels.index(color_type_group[0])
            type_ = type_labels.index(color_type_group[1])
            self.vehicle_colors.append(np.float32(color_))
            self.vehicle_types.append(np.float32(type_))
            self.images.append(os.path.join(root_dir, file_name))
            index += 1

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]
        img = cv.imread(image_path)  # BGR order
        # h, w, c = img.shape
        # rescale
        # img = cv.resize(img, (72, 72))
        # img = (np.float32(img) /255.0 - 0.5) / 0.5
        # H, W C to C, H, W
        # img = img.transpose((2, 0, 1))
        sample = {'image': self.transform(img), 'color': self.vehicle_colors[idx], 'type': self.vehicle_types[idx]}
        return sample


if __name__ == "__main__":
    ds = VehicleAttrsDataset("/Users/remilia/Documents/02-Work/05-Python/0-CustomDb/vehicle_attrs_dataset/")
    print(len(ds))
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['color'])
        if i == 3:
            break

    # dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['type'])
        break
