import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2 as cv


class FaceLandmarksDataset(Dataset):
    def __init__(self, txt_file):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                  std=[0.5, 0.5, 0.5]),
                                             transforms.Resize((64, 64))
                                             ])
        lines = []
        with open(txt_file) as read_file:
            for line in read_file:
                line = line.replace('\n', '')
                lines.append(line)
        self.landmarks_frame = lines

    def __len__(self):
        return len(self.landmarks_frame)

    def num_of_samples(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        contents = self.landmarks_frame[idx].split('\t')
        image_path = contents[0]
        img = cv.imread(image_path)  # BGR order
        h, w, c = img.shape
        # rescale
        # img = cv.resize(img, (64, 64))
        # img = (np.float32(img) /255.0 - 0.5) / 0.5
        landmarks = np.zeros(10, dtype=np.float32)
        for i in range(1, len(contents), 2):
            landmarks[i - 1] = np.float32(contents[i]) / w
            landmarks[i] = np.float32(contents[i + 1]) / h
        landmarks = landmarks.astype('float32').reshape(-1, 2)
        # H, W C to C, H, W
        # img = img.transpose((2, 0, 1))

        sample = {'image': self.transform(img), 'landmarks': torch.from_numpy(landmarks)}
        return sample


if __name__ == "__main__":
    ds = FaceLandmarksDataset("D:/facedb/Face-Annotation-Tool/landmark_output.txt")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['landmarks'].size())
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())
