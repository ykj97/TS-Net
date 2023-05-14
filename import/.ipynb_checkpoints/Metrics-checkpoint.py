#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSC(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DSC, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        smooth = 1.0
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice =  (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice
    
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        smooth = 1.0
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice =  (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        dice_loss = 1 - dice
        
        return dice_loss
    
# https://www.kaggle.com/ligtfeather/semantic-segmentation-is-easy-with-pytorch
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        ## Fix
        mask = mask.squeeze()
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy    


# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
ALPHA = 0.75
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
    
    
    
## Output저장하기 위한 필요한 function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1) # tensor variable에서 numpy로 transfer 시키는 함수
fn_denorm = lambda x, mean, std: (x*std) + mean # denormalization    # classification() using thresholding (p=0.5)
fn_class = lambda x: 1.0 * (x > 0.5)  # 네트워크 output의 img를 binary class로 분류해주는 function 
