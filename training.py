import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model import generator, loss
from PIL import Image



class LandScapeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.input_images = os.listdir(os.path.join(root_dir, 'InputImages'))
        self.gt_images = os.listdir(os.path.join(root_dir, 'GroundTruthImages'))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img_name = os.path.join(self.root_dir, 'InputImages', self.input_images[idx])
        gt_img_name = os.path.join(self.root_dir, 'GroundTruthImages', self.gt_images[idx])
        mask_image = os.path.join(self.root_dir, 'Mask', self.gt_images[idx])

        input_image = Image.open(input_img_name).convert("RGB")
        gt_image = Image.open(gt_img_name).convert("RGB")
        mask_image = Image.open(mask_image).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
            mask_image = self.transform(mask_image)

        return input_image, gt_image, mask_image


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset = LandScapeDataset(root_dir='dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = generator.generator()
criterion = loss.ModifiedEuclideanLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1

def train():
    for epoch in range(num_epochs):

        for batch_idx, (input_image, gt_image, mask_image) in enumerate(data_loader):
            # optimizer.zero_grad()
            x, y = model(input_image, mask_image)
            return print(x.shape, y.shape)
            # loss = criterion(output, gt_image)
            # loss.backward()
            # optimizer.step()

            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(input_image), len(data_loader.dataset),
            #         100. * batch_idx / len(data_loader), loss.item()))
    


if __name__ == '__main__':
    train()


