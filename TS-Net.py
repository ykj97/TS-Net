import torch
print('Graphic name:', torch.cuda.get_device_name())

import sys
sys.path.insert(0, "import/")
from util import *
from Metrics import *
from dataset_albu import *
from Unet import *

import argparse 
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import tqdm
import torch.nn.functional as F
import torch.optim.lr_scheduler 
import segmentation_models_pytorch as smp

import gc
gc.collect()
torch.cuda.empty_cache()

## parameter and directory path
lr = 3e-04
batch_size = 16
test_data_dir = 'data/testset/'
fold = ['fold1','fold2','fold3','fold4','fold5']
test_folder = ['treated_subject_1', 'untreated_subject_1']
mode = "test"
train_continue = "off"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nParamterter setting")
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("mode: %s" % mode)
print("train_continue : %s" % train_continue)

test_transforms = A.Compose([A.Normalize(mean=0.5, std=0.5),
                            ToTensorV2(transpose_mask=True)])
TS_Net = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights="imagenet",
    activation='sigmoid',
    in_channels=1,
    classes=1)

net = TS_Net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
loss = smp.losses.FocalLoss(mode='binary', alpha=0.75, gamma=2)
Dice = DSC().to(device)

total_DSC_arr = []
total_iou_arr = []

for itr, test_name in enumerate(test_folder):
    print("\nEvaluate TS-Net with %s" %test_name)
    
    for num_fold in fold:
        data_dir = 'data/testset/' + num_fold
        ckpt_dir = 'ckpt/' + num_fold
        result_ensemble_dir = 'result/'
        
        if not os.path.exists(result_ensemble_dir):
            folders = ['treated_subject_1', 'untreated_subject_1']
            for folder in folders:
                os.makedirs(os.path.join(result_ensemble_dir, folder))  
                
        dataset_test = Dataset(data_dir=os.path.join(test_data_dir,test_name), transform=test_transforms)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)
                               
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

        preds = []

        if mode == 'test':

            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

            output_arr_fold = []
            dsc_score_arr = []
            iou_score_arr = []
            f1_score_arr = []

            with torch.no_grad():
                net.eval()

                for batch, data in enumerate(loader_test, 1):

                    label = data['label'].to(device)
                    input = data['input'].to(device)

                    input -= input.min()
                    input /= input.max()

                    label = label / 255.0

                    output = net(input)

                    output = fn_tonumpy(output)

                    output_arr_fold.append(output)
                                  
                               
        if num_fold == 'fold1':
            output_arr_fold1 = output_arr_fold
            print("\n---- Save successfully output data of %s" % num_fold)
                    
        elif num_fold == 'fold2':
            output_arr_fold2  = output_arr_fold
            print("---- Save successfully output data of %s" % num_fold)

        elif num_fold == 'fold3':
            output_arr_fold3  = output_arr_fold
            print("---- Save successfully output data of %s" % num_fold)

        elif num_fold == 'fold4':
            output_arr_fold4  = output_arr_fold
            print("---- Save successfully output data of %s" % num_fold)

        elif num_fold == 'fold5':
            output_arr_fold5  = output_arr_fold
            print("---- Save successfully output data of %s" % num_fold)
    
    
    print("---- Test all folds... time to ensemble")      
        
    dsc_score_arr = []
    iou_score_arr = []
    preds_arr = []
    
                               
    for batch, data in enumerate(loader_test, 1):
        num = batch-1
              
        preds = (output_arr_fold1[num] + output_arr_fold2[num] + output_arr_fold3[num] 
                    + output_arr_fold4[num] + output_arr_fold5[num]) / 5
        
                               
        fn_class = lambda x: 1.0 * (x > 0.4)   
        preds = fn_class(preds)
        preds_arr.append(preds)

        output = np.transpose(preds, (0,3,1,2))
        plt.imsave(os.path.join(result_ensemble_dir, test_name, '%s_%04d.png' % (test_name, batch)), preds.squeeze(),cmap='gray')

        output = torch.Tensor(output).to(device)
        
        label = data['label'].to(device)
        label = label / 255.0
            
        dsc_score = Dice(output, label)
        dsc_score = dsc_score.to('cpu').detach().numpy()
        dsc_score_arr += [dsc_score.item()]
        
        label = label.to(torch.int32)

        tp, fp, fn, tn = smp.metrics.get_stats(output, label, mode='binary', threshold=0.5);
            
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        iou_score = iou_score.to('cpu').detach().numpy()   
        iou_score_arr += [iou_score.item()]
        
    
    print("---- AVG DSC : %.4f | AVG IoU : %.4f" 
            % (np.mean(dsc_score_arr), np.mean(iou_score_arr)))
    
    total_DSC_arr += [np.mean(dsc_score_arr).item()]
    total_iou_arr += [np.mean(iou_score_arr).item()]
    
    if test_name == 'treated_subject_1':
        enesemble_C02 = preds_arr
        print("---- Save successfully enesemble output data of %s" % test_name)
    elif test_name == 'untreated_subject_1':
        enesemble_C10 = preds_arr
        print("---- Save successfully enesemble output data of %s" % test_name)
        
print("\nTS-Net AVG Dice : %.3f (SD : %.3f) | AVG IoU : %.3f (SD : %.3f)" 
      % (np.mean(total_DSC_arr),np.std(total_DSC_arr),np.mean(total_iou_arr),np.std(total_iou_arr)))

