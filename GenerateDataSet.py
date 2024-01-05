# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:26:45 2021

@author: lxw
"""
#import torch
#import torch.utils.data
#import random
#from torch import optim, nn
#from torchvision import datasets
#from torchvision.transforms import transforms
#import torchvision
#import torch.nn.functional as F
from torch.utils.data import DataLoader
#from Nets import MLP,RNN
#import scipy.io as scio
#import h5py
#import numpy as np
from osgeo import gdal
#from osgeo import ogr
#from PIL import Image
from MyUtils import RemoteSensingDataset
from config import configs

def load_remote_sensing_train_dataset(image_path, point_patch):

    print("Image path:", image_path)
    print("Point path:", point_patch)
    train_dataset = RemoteSensingDataset(image_path,point_patch)
#    print(dataset.bands_num, dataset.image_height, dataset.image_width, dataset.point_num)
    
    dataloader = DataLoader( train_dataset,  batch_size = 1, shuffle=False)
    
    for i_batch,data in enumerate(dataloader):
#        print(i_batch)
        index, label, inner_scale, obj_scale, scene_scale = data      
#        print(scene_scale.shape)
#        img = scene_scale[0]
#        im = Image.fromarray(np.asarra(img))
#        im.save("t1.jpg")
        print(index, label, inner_scale.shape, obj_scale.shape, scene_scale.shape)
        break
    
def main(argv = None):
    gdal.AllRegister()    
    load_remote_sensing_train_dataset(configs.image_path,configs.points_path)
    
    print ("Finished!")
    
if __name__ == '__main__':
    main()    
