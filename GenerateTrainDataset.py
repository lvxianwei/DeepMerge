# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:32:26 2021

@author: lxw
"""

import numpy as np
#import os
import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import transforms
#from torchvision.utils import make_grid
# from skimage import io, transform
# import h5py
import random
#import numpy as np
from osgeo import gdal
from osgeo import ogr
#from PIL import Image
from config import configs
import cv2

    
class RemoteSensingDataset():
    
    def __init__(self, point_path, traing_dataset_txt_path, train_num = 0):
        gdal.AllRegister()
        self.point_dataset, self.point_list, self.point_dataset_length, self.classes = self.open_vector_as_ds_and_dic(point_path)
        self.class_name_list = []
        for key in self.classes:
            self.class_name_list.append(key)
        self.class_num = len(self.classes)    
        # self.point_layer = self.point_dataset.GetLayer(0)
        self.train_num = train_num
        self.traing_dataset_txt_path = traing_dataset_txt_path
    
    def GenerateTrainDataset(self):
        train_dataset_txt = open(self.traing_dataset_txt_path, 'w')   
        i = 0
        # print(self.train_num)
        while i < self.train_num:
            classes_range = range(0, self.class_num)
#        for i in range (0, self.dataset_length):
            positive_label, negative_label = random.sample(classes_range,2)
            anchor_label = positive_label
            name_positive = self.class_name_list[positive_label]
            name_negative = self.class_name_list[negative_label]
                
            class_positive_num =len(self.classes[name_positive])
            class_negative_num = len(self.classes[name_negative])
            
            positive_range = range(0, class_positive_num)
            negative_range = range(0, class_negative_num)
                
            positive_id_in_dict, anchor_id_in_dict = random.sample(positive_range,2)
            negative_id_in_dict = random.sample(negative_range,1)[0]
            
            positive_id = self.classes[name_positive][positive_id_in_dict]
            anchor_id   = self.classes[name_positive][anchor_id_in_dict]
            negative_id = self.classes[name_negative][negative_id_in_dict]
            i +=1
#           print(positive_id, anchor_id, negative_id)
            train_dataset_txt.write("{0}:{1},{2},{3},{4},{5},{6}\n".format(i, positive_label, anchor_label, negative_label, positive_id, anchor_id, negative_id))
        train_dataset_txt.close()
        # return positive_label, anchor_label, negative_label, positive_id, anchor_id, negative_id
        print("Genetate sucessfully!")
    def open_vector_as_ds_and_dic(self, shapefile_path, is_random = False):  
        vector_DataSource, layer, num_features = self.open_vector_as_ds(shapefile_path)
        temp_list = []
        class_dictionary = {}
        
        for num in range(0,num_features):
            feature = layer.GetFeature(num)
            class_name = feature.GetField("class_name")
            if class_dictionary.__contains__(class_name) == False:
#                print(class_name)
                class_dictionary[class_name] = []
                class_dictionary[class_name].append(num)
            else:
                class_dictionary[class_name].append(num)
            temp_list.append(num)                                    
        if is_random:
            random.seed(10) #设定随机种子
            random.shuffle(temp_list) #随机打乱图像列表  
            for key in class_dictionary:
                random.shuffle(class_dictionary[key])
        layer.ResetReading()
        return vector_DataSource, temp_list, num_features, class_dictionary
    
    def open_vector_as_ds(self, shapefile_path):
        vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
        vector_DataSource = vector_deriver.Open(shapefile_path,0)  
        if vector_DataSource is None:
            raise ValueError("Can not open {0}".format(shapefile_path))
        layer = vector_DataSource.GetLayer(0)
        num_features = layer.GetFeatureCount()
        return vector_DataSource, layer, num_features

def main():
    train_ds = RemoteSensingDataset(configs.points_path, configs.traing_dataset_txt_path + "train_dataset.txt", train_num=20000 )         
    train_ds.GenerateTrainDataset()
    return
        
        
if __name__ == "__main__":
    main()