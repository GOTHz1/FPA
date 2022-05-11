import os

import numpy as np
import cv2
import sys

import torch
from torchvision import transforms
from tools import utils

sys.path.append('..')
import math
from torch.utils import data
from torch.utils.data import DataLoader


class Datasets(data.Dataset):
    def __init__(self, file_list, landmark_size, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.filenames = None
        self.euler_angle = None
        self.R = None
        self.landmark_size = landmark_size
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])

        if self.landmark_size == 68:
            self.landmark = np.asarray(self.line[1:137], dtype=np.float32)
        elif self.landmark_size == 32:
            self.landmark = np.asarray(self.line[73:137], dtype=np.float32)
        self.euler_angle = np.asarray(self.line[137:140], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        self.R = utils.get_R(self.euler_angle[0], self.euler_angle[1], self.euler_angle[2])
        return (self.img, self.landmark, self.euler_angle, torch.FloatTensor(self.R))

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = '300WLP/lists.txt'
    img_size = 224
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size)])

    datasets = Datasets(file_list, 68, transform)
    dataloader = DataLoader(datasets,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)
    for img, landmark, pose_angle, R_pred in dataloader:

        img = np.array(
            np.transpose(img[0].cpu().numpy(), (1, 2, 0)))

        img = (img * 255).astype(np.uint8).copy()
        landmar = np.array(landmark[0] * img_size).astype("int16").reshape(-1, 2)

        for lan in landmar:
            cv2.circle(img, lan, 0, [255, 0, 0])

        pose = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi


        pose = np.array(pose[0].cpu().numpy()).astype("float")
        x = math.radians(pose[0])
        y = math.radians(pose[1])
        z = -math.radians(pose[2])
        utils.showimgFromeuler(img, x, y, z, landmark, 'euler')
        cv2.waitKey(0)
