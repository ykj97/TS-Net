import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from PIL import Image
import PIL.Image as pilimg

## 데이터 로더 구현하기

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')] # startswitch : 문자열이 특정문자로 시작하는지 여부를 알려준다
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        

        # 바꾼부분
        #label = label / 255.0 # 내 label은 0-1로 되어있는데 이걸 해줘야하나
        #input = input / 255.0

        # Neural network에 들어가는 모든 input은 3개의 axis를 가지고 있어야 함(x,y,channel)
        
        if label.ndim == 2:
            label = label[:, :, np.newaxis] #2개면 새로운 axis 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        #data = {'input': input, 'label': label} # 딕셔너리 형태
        
        if self.transform:

            
            data = self.transform(image=input, mask=label)

            data_img = data["image"] ## 수정
            data_lbl = data["mask"]
        
            
            data = {'input': data_img, 'label': data_lbl}
            
        return data

"""

---





---

"""

## Data transform 함수 구현하기

class ToTensor(object): # 필수적으로 들어가야함 numpy -> tensor
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # Image의 numpy 차원 = (Y,X,CH) / Image의 tensor 차원 = (CH,Y,X) -> dimension의 순서가 다름
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5): #std = standard deviation
        self.mean = mean
        self.std = std

    def __call__(self, data): # 실제로 쓰이는 함수
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): 
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label) # flip left right
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label) # flip up and down
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data


