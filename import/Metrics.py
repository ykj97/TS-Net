
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
    

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        ## Fix
        mask = mask.squeeze()
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy    


ALPHA = 0.75
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        

        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
    
    
    
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1) 
fn_denorm = lambda x, mean, std: (x*std) + mean 
fn_class = lambda x: 1.0 * (x > 0.5) 
