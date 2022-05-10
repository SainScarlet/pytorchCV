import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv


class FaceLandmarksDataset(Dataset):
    def __init__(self, txt_file):
        # transform会自动把cv中imread读到的HWC图像转换为CHW的shape，方便torch导入
        # 不用transform的化，特别需要注意在cv中要执行img = img.transpose((2, 0, 1))
        # 注意需要在resize之前加一个transforms.ToPILImage()，一定要注意这几个函数的顺序
        # 关于使用GPU使用，在我们想把 GPU tensor 转换成 Numpy 变量的时候，需要先将 tensor 转换到 CPU 中去
        # 因为Numpy 是 CPU-only 的
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 读取每一行数据
        lines = []
        with open(txt_file) as read_file:
            for line in read_file:
                line = line.replace('\n', '')
                lines.append(line)
        self.landmarks_frame = lines

    # 样本数
    def __len__(self):
        return len(self.landmarks_frame)

    def num_of_samples(self):
        return len(self.landmarks_frame)

    # 数据集 map.style 的数据集就是重现getitem
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        contents = self.landmarks_frame[idx].split('\t')
        # 第一列加载图像位置
        image_path = contents[0]
        img = cv.imread(image_path)  # BGR order
        h, w, c = img.shape
        # 同上面的transform处理图像
        # img = cv.resize(img, (64, 64))
        # img = (np.float32(img) /255.0 - 0.5) / 0.5

        # landmark的5个点对应10个值
        landmarks = np.zeros(10, dtype=np.float32)
        # 把真实坐标处理成0-1的值
        for i in range(1, len(contents), 2):
            landmarks[i - 1] = np.float32(contents[i]) / w
            landmarks[i] = np.float32(contents[i + 1]) / h

        # 坐标reshape一个5*2的矩阵
        landmarks = landmarks.astype('float32').reshape(-1, 2)
        # 不用transform，CV中调换为CHW
        # H, W C to C, H, W
        # img = img.transpose((2, 0, 1))

        # 用transform处理，归一化，0.5偏移，0中心化，resize，转换为tensor
        sample = {'image': self.transform(img), 'landmarks': torch.from_numpy(landmarks)}
        return sample


if __name__ == "__main__":
    ds = FaceLandmarksDataset(
        "/Users/remilia/Documents/02-Work/05-Python/0-CustomDb/landmark_dataset/"
        "landmark_output.txt"
    )
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['landmarks'].size())
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())
