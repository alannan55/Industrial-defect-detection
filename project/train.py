import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score
from PIL import Image
from tqdm import tqdm
from config import CFG
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, ViTForImageClassification


# Load data
label_data = pd.read_csv(os.path.join(CFG.train_data_dir, '文件标签汇总数据.csv'))
train_csv_folder = os.path.join(CFG.train_data_dir, 'csv文件')
test_A_csv_folder = os.path.join(CFG.test_A_data_dir, 'csv文件')
train_image_folder = os.path.join(CFG.project_name, 'project/image/训练集数据')
test_A_image_folder = os.path.join(CFG.project_name, 'project/image/A榜测试集数据')


class TrainDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            image_folder) if f.endswith('.png')]
        # 假设标签以某种方式存储或从文件名中获取

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = CFG.label2id[self.image_files[idx][0]]

        if self.transform:
            image = self.transform(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = TrainDataset(
    image_folder=train_image_folder, transform=data_transforms['train'])
test_dataset = TestDataset(
    image_folder=test_A_image_folder, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=4,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4,
                         shuffle=False, num_workers=4)

dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}


# 加载预训练的vit模型
model = ViTForImageClassification.from_pretrained('/ai/users/bst/competition/model/vit-base-patch16-224',
                                                  num_labels=5,
                                                  ignore_mismatched_sizes=True
                                                  )

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / dataset_sizes['train']
    print(f'Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), os.path.join(
    CFG.project_name, 'project/model/model.pth'))
