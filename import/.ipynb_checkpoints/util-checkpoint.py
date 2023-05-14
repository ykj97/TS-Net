import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net']) 
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
    
def configurate(lr, epoch, batch_size):
    print("learning rate : %.4e" % lr)
    print("number of epoch : %d" % epoch)
    print("batch size : %d" % batch_size)
    return lr, epoch, batch_size

def path(data_dir, ckpt_dir, log_dir, test_data_dir, result_ensemble_dir):

    
    print("data dir : %s" % data_dir)
    print("ckpt dir : %s" % ckpt_dir)
    print("log dir : %s" % log_dir)
    print("test_data_dir : %s" % test_data_dir)
    print("result_ensemble_dir : %s " % result_ensemble_dir)
    
    return data_dir, ckpt_dir, log_dir, test_data_dir, result_ensemble_dir


def mode(mode, train_continue):
    print("mode : %s" % mode)
    print("train_continue : %s" % train_continue)
    
    return(mode, train_continue)


