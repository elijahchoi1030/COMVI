import csv
import imghdr
from turtle import forward
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from PIL import Image as pilimg
import os
import numpy as np
import random

class myModelBlock(nn.Module):
  def __init__(self, input_size, channel_size, first_stride):
    super().__init__()
    self.stride = first_stride
    self.block = nn.Sequential(
      nn.Conv2d(input_size, channel_size[0], kernel_size=1, padding=0, stride=first_stride),
      nn.BatchNorm2d(channel_size[0]),
      nn.ReLU(),
      nn.Conv2d(channel_size[0], channel_size[1], kernel_size=3, padding=1, stride=1),
      nn.BatchNorm2d(channel_size[1]),
      nn.ReLU(),
      nn.Conv2d(channel_size[1], channel_size[2], kernel_size=1, padding=0, stride=1),
      nn.BatchNorm2d(channel_size[2]),
    )
    if self.stride == 2:
      self.strider = nn.Sequential(
        nn.Conv2d(input_size, channel_size[2], kernel_size=1, padding=0, stride=2),
        nn.BatchNorm2d(channel_size[2]),
      )

  def forward(self, x):
    origin = x
    y = self.block(x)
    if self.stride == 2:
      origin = self.strider(x)
    y += origin
    return nn.functional.relu(y)

class MyModel(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            self.repeatable_block(64, [64, 64, 256], repeats=3)
        )
        self.conv3 = self.repeatable_block(256, [128, 128, 512], repeats=4)
        self.conv4 = self.repeatable_block(512, [256, 256, 1024], repeats=6)
        self.conv5 = self.repeatable_block(1024, [512, 512, 2048], repeats=3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, out_channels)

    def repeatable_block(self, input_size, channel_size, repeats=1):
        stride = 2
        block = []
        for i in range(repeats):
            block.append(myModelBlock(input_size, channel_size, stride))
            input_size = channel_size[2]
            stride = 1
        return nn.Sequential(*block)

    def forward(self,x) :
        #TODO:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class MyDataset(Dataset) :
    def __init__(self, meta_path, root_dir, transform=None) :
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.img_normal = TorchvisionNormalize()
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        if self.transform != 'train':
            self.len_data = 959

    def __len__(self):
        if self.transform == 'train':
            return len(self.metadata['annotations'])
        else:
            return self.len_data

    def __getitem__(self, idx):
        if self.transform == 'train':
            img_name = self.metadata['annotations'][idx]['file_name']
            img_label = self.metadata['annotations'][idx]['category']
        else:
            img_name = f"{idx}.jpg"
        img_path = os.path.join(self.root_dir, img_name)

        img = pilimg.open(img_path).convert('RGB')
        img = np.array(img)

        # normalization
        img = self.img_normal(img)

        # flip
        if self.transform == 'train':
            if bool(random.getrandbits(1)):
                img = np.fliplr(img)

        # center crop
        H, W = img.shape[0:2]
        if H > W:
            if self.transform == 'train':
                randnum = int(np.random.triangular(0, (H-W)/2, H-W+1))
            else:
                randnum = (H-W)//2
            img = img[randnum:randnum + W,:,:]
        else:
            if self.transform == 'train':
                randnum = int(np.random.triangular(0, (W-H)/2, W-H+1))
            else:
                randnum = (W-H)//2
            img = img[:,randnum:randnum + H,:]


        # resize to 224 by 224
        img = np.asarray(pilimg.fromarray(img, mode='RGB').resize((224, 224)))
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        if self.transform == 'train':
            return np.float32(img), np.int64(img_label)
        else:
            return np.float32(img), img_name


def train(model, device, epochs, train_dataloader) :
    #TODO: Make your own training code
    num_iter = 40000//train_dataloader.batch_size
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-03)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
        milestones=[15*num_iter, 30*num_iter, 45*num_iter, 60*num_iter], gamma=0.5)

    avg_losses = []
    train_acc = []

    for epoch in range(epochs):
        loss_sum = 0.0
        correct = 0
        total = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss

            _, preds = torch.max(outputs, 1)
            total += img.shape[0]
            correct += (preds == label).sum().item()
        avg_loss = loss_sum / len(train_dataloader)
        avg_losses.append(avg_loss)
        accuracy = 100 * correct / total
        train_acc.append(accuracy)
        
        print(f'(Train) [epoch : {epoch+1:.3f}] loss: {avg_loss:.3f} | acc: {accuracy:.3f} | lr: {optimizer.param_groups[0]["lr"]}')
        torch.save(model.state_dict(), './checkpoint.pth') 

    print('Training Ended')

    return model


    # You SHOULD save your model by
    # torch.save(model.state_dict(), './checkpoint.pth') 
    # You SHOULD not modify the save path
    pass


def get_model(model_name, checkpoint_path):
    
    model = model_name(3, 80)
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def test():
    
    model_name = MyModel
    checkpoint_path = './model.pth' 
    mode = 'test' 
    data_dir = "./test_data"
    meta_path = "./answer.json"
    batch_size = 8
    model = get_model(model_name, checkpoint_path)

    data_transforms = {
        'train' :"YOUR_DATA_TRANSFORM_FUNCTION" , 
        'test': "YOUR_DATA_TRANSFORM_FUNCTION"
    }

    # Create training and validation datasets
    test_datasets = MyDataset(meta_path, data_dir, data_transforms[mode])

    # Create training and validation dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Set model as evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Inference
    result = []
    for images, filename in tqdm(test_dataloader):
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(num_image):
            result.append({
                'filename': filename[i],
                'class': preds[i].item()
            })

    result = sorted(result,key=lambda x : int(x['filename'].split('.')[0]))
    
    # Save to csv
    with open('./result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','class'])
        for res in result:
            writer.writerow([res['filename'], res['class']])


def main() :    
    test()
    

if __name__ == '__main__':
    main()