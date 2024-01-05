# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:04:25 2021

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
import random
#import numpy as np
from osgeo import gdal
from osgeo import ogr
#from PIL import Image
from config import configs
import cv2
    
class RemoteSensingDataset(Dataset):
    
    def __init__(self, image_path, point_path, is_train = False, train_num = 0, polygon_path = None):
        gdal.AllRegister()
        self.image_dataset = self.open_image_as_dataset(image_path)
        self.point_dataset, self.point_list, self.point_dataset_length, self.classes = self.open_vector_as_ds_and_dic(point_path)
        self.is_train = is_train
        if self.is_train == True:
            self.train_num = train_num
            if self.classes.__contains__("unclassified"):
                del self.classes["unclassified"]
            for key in self.classes:
               print(key)
        else:
            if polygon_path == None:
                raise ValueError('Invalid input arguments! there is no polygon_path.')
            self.polygon_dataset, self.polygon_layer, self.polygon_dataset_length = self.open_vector_as_ds(polygon_path)
            print("test data length:",self.polygon_dataset_length)
        self.class_name_list = []
        for key in self.classes:
            self.class_name_list.append(key)
        self.class_num = len(self.classes)    
        self.bands_num = self.image_dataset.RasterCount
        self.image_width = self.image_dataset.RasterXSize#源图像宽
        self.image_height=self.image_dataset.RasterYSize
        self.point_layer = self.point_dataset.GetLayer(0)
        train_file = open(configs.traing_dataset_txt_path + "train_dataset.txt",'r')
        self.train_data=train_file.readlines()
        train_file.close()
        
        
    def __len__(self):
        
        if self.is_train == True:
            # return len(self.train_data)
            return self.train_num
        else:
            return self.polygon_dataset_length
    
    def __getitem__(self, index):

        if self.is_train == True:
            return self.GenerateTrainDataset(index)
        else:
            return self.GenerateTestDataset(index)

    
    def GenerateTrainDataset(self, index):
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
        
#        print(positive_id, anchor_id, negative_id)
        
   
        # line = self.train_data[index]
        # line = line[:-1].split(':')[-1].split(',')
        # line = [int(x) for x in line]
        # positive_label, anchor_label, negative_label, positive_id, anchor_id, negative_id = line


        positive_data = self.GetDataByID(positive_id)
        pos_inner, pos_obj, pos_scene = positive_data
        anchor_data = self.GetDataByID(anchor_id)
        anc_inner, anc_obj, anc_scene = anchor_data
        negative_data = self.GetDataByID(negative_id)
        neg_inner, neg_obj, neg_scene = negative_data
        
        return positive_label, anchor_label, negative_label, pos_inner, pos_obj, pos_scene, anc_inner, anc_obj, anc_scene , neg_inner, neg_obj, neg_scene
        # return positive_label, anchor_label, negative_label, pos_inner, pos_obj, pos_scene, anc_inner, anc_obj, anc_scene , neg_inner, neg_obj, neg_scene
    
    def GenerateTestDataset(self,index):
        object_feature  = self.polygon_layer.GetFeature(index)
        neighbors= str(object_feature.GetField("join")).split(',')
        neighbors = [int(x) for x in neighbors ]
        neighbors.remove(index) #相邻对象的id
        
        object_points = str(object_feature.GetField("Points")).split(',')
        object_points = [int(x) for x in object_points]
        
#        img_w = configs.target_img_size
#        img_h = configs.target_img_size
#        print(object_points)
        object_data_array = np.zeros([])
        for i in range(0, len(object_points)):
            pt_id = object_points[i]
#            print(pt_id)
            data1, data2, data3 = self.GetDataByID(pt_id)
#        print(data1.shape)
            data1 = data1[np.newaxis,:,:,:]
            data2 = data2[np.newaxis,:,:,:]
            data3 = data3[np.newaxis,:,:,:]
            object_data = data1
            object_data = np.concatenate((object_data,data2), axis = 0)
            object_data = np.concatenate((object_data,data3), axis = 0)
            object_data = object_data[np.newaxis,:,:,:,:]
            if i == 0:
                object_data_array = object_data
            else:
                object_data_array = np.concatenate((object_data_array,object_data), axis = 0)
#            print("第",i,"个点数据：",object_data_array.shape)
            

        
        neighbor_datas = np.zeros([])
        
        for i in range(0, len(neighbors)):
#            idx = self.point_layer.GetFeature()
#            self.GetDataByID(i)            
#            print(neighbors[i])
            #这里的neighbors[i]是面对象，应该根据面对向搜索点对象
            neighbor_feature = self.polygon_layer.GetFeature(neighbors[i])
            neighbor_points = str(neighbor_feature.GetField("Points")).split(',')
            neighbor_points = [int(x) for x in neighbor_points]
            
            neighbor_data_array = np.zeros([])
            for j in range(0, len(neighbor_points)):
                pt_id = neighbor_points[j]
                data1, data2, data3 = self.GetDataByID(pt_id)
                data1 = data1[np.newaxis,:,:,:]
                data2 = data2[np.newaxis,:,:,:]
                data3 = data3[np.newaxis,:,:,:]
                neighbor_data = data1
                neighbor_data = np.concatenate((neighbor_data,data2), axis = 0)
                neighbor_data = np.concatenate((neighbor_data,data3), axis = 0)
                neighbor_data = neighbor_data[np.newaxis,:,:,:,:]                
                if j == 0:
                    neighbor_data_array = neighbor_data
                else:
                    neighbor_data_array = np.concatenate((neighbor_data_array, neighbor_data), axis = 0)
#                print("第",j,"个点数据：",neighbor_data_array.shape)
            neighbor_data_array = neighbor_data_array[np.newaxis,:,:,:,:,:]
            if i == 0:
                neighbor_datas = neighbor_data_array
            else:
                neighbor_datas = np.concatenate((neighbor_datas, neighbor_data_array), axis = 0)
#            print("第",i,"个对象数据：",neighbor_datas.shape)
#            neighbor_data1, neighbor_data2, neighbor_data3 = self.GetDataByID(neighbors[i])
#            neighbor_data1 = neighbor_data1[np.newaxis,:,:,:]
#            neighbor_data2 = neighbor_data2[np.newaxis,:,:,:]
#            neighbor_data3 = neighbor_data3[np.newaxis,:,:,:]
#            
#            neighbor_data = neighbor_data1
#            neighbor_data = np.concatenate((neighbor_data,neighbor_data2), axis = 0)
#            neighbor_data = np.concatenate((neighbor_data,neighbor_data3), axis = 0)
#            neighbor_data = neighbor_data[np.newaxis,:,:,:,:]
#            if i == 0:
#                neighbor_datas = neighbor_data
#            else:
#                neighbor_datas = np.concatenate((neighbor_datas,neighbor_data),axis = 0)
#        print(neighbor_datas.shape)
#        print(neighbors)
        
        neighbor_indexes = neighbors
        obj_index = index
        return object_data_array, neighbor_datas,obj_index,neighbor_indexes
    
    def GetDataByID(self, index):
        
        feature = self.point_layer.GetFeature(index)
        
        inner_scale = int(feature.GetField("inner"))
        obj_scale   = int(feature.GetField("object"))
        scene_scale = int(feature.GetField("scene"))
        geometry = feature.GetGeometryRef()
        XGeo = geometry.GetX()
        YGeo = geometry.GetY()
        
        geoTrans = self.image_dataset.GetGeoTransform()
        XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
        YLine = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
        
        inner_points = self.calculate_left_top_point_and_size(XPixel,YLine,inner_scale)
        obj_points   = self.calculate_left_top_point_and_size(XPixel,YLine,obj_scale)
        scene_points = self.calculate_left_top_point_and_size(XPixel,YLine,scene_scale)
        
        inner_region = self.cut_image(self.image_dataset,inner_points)
        obj_region   = self.cut_image(self.image_dataset,obj_points)
        scene_region = self.cut_image(self.image_dataset,scene_points)

        inner_region = inner_region.transpose(1,2,0)
        obj_region   = obj_region.transpose(1,2,0)
        scene_region = scene_region.transpose(1,2,0)
     
        img_w = configs.target_img_size
        img_h = configs.target_img_size
        inner_img = np.zeros([img_w, img_h, 1])
        obj_img = np.zeros([img_w, img_h, 1])
        scene_img = np.zeros([img_w, img_h, 1])
        for i in range(0, self.bands_num):
            inner_band = inner_region[:,:,i]
            inner_band = cv2.resize(np.array(inner_band), (img_h, img_w),interpolation=cv2.INTER_AREA)
            inner_band = inner_band[:,:, np.newaxis]
            
            obj_band = obj_region[:,:,i]
            obj_band = cv2.resize(np.array(obj_band), (img_h, img_w),interpolation=cv2.INTER_AREA)
            obj_band = obj_band[:,:,np.newaxis]
            
            scene_band = scene_region[:,:,i]
            scene_band = cv2.resize(np.array(scene_band), (img_h, img_w),interpolation=cv2.INTER_AREA)
            scene_band = scene_band[:,:,np.newaxis]
            
            if i == 0:
                inner_img = inner_band
                obj_img   = obj_band
                scene_img = scene_band
            else:
                inner_img = np.concatenate((inner_img, inner_band),axis = 2)
                obj_img   = np.concatenate((obj_img, obj_band),axis = 2)
                scene_img = np.concatenate((scene_img, scene_band),axis = 2)
                
                
        inner_region = inner_img.transpose(2,0,1)
        obj_region = obj_img.transpose(2,0,1)
        scene_region = scene_img.transpose(2,0,1)
        
#        if self.transform is not None:
#            inner_region = self.transform(inner_region)
#            obj_region   = self.transform(obj_region)
#            scene_region = self.transform(scene_region)
        inner_region = inner_region.astype(np.float32) / 255.0
        obj_region   = obj_region.astype(np.float32) / 255.0
        scene_region = scene_region.astype(np.float32) / 255.0
#        print(inner_region.shape)
#        print(obj_region.shape)
#        print(scene_region.shape)
#        print("shape",scene_region.shape)
#        im = Image.fromarray(scene_region)
#        im.save("t.jpg")
        return inner_region, obj_region, scene_region    
    def open_image_as_dataset(self, image_path):
        ds = gdal.Open(image_path,gdal.GA_ReadOnly)
        if ds == None:
#            print("cannot open", image_path)
            raise ValueError("Can not open {0}".format(image_path))
#            return
        return ds
    
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
        
    
    def cut_image(self,image_dataset,list_points):
        startOrgX=list_points[0]
        startOrgY=list_points[1]
        dstWidth=list_points[2]
        dstHeight=list_points[3]
        bandsNum = image_dataset.RasterCount
        srcWidth = image_dataset.RasterXSize#源图像宽
        srcHeight=image_dataset.RasterYSize
        outputWidth = dstWidth
        outputHeight = dstHeight
        startX = startOrgX
        outputStartX = 0
        if (startX < 0):
            dstWidth += startX
            outputStartX = -startX
            startX = 0
        if (startX + dstWidth >= srcWidth):
            dstWidth = srcWidth - startX
        startY = startOrgY
        outputStartY = 0
        if (startY < 0):        
            dstHeight += startY
            outputStartY = -startY
            startY = 0
        if (startY + dstHeight >= srcHeight):       
            dstHeight = srcHeight - startY
        databuf = image_dataset.ReadAsArray(int(startX),int(startY),int(dstWidth), int(dstHeight))
        dst=np.zeros((bandsNum,outputHeight,outputWidth),dtype=np.uint8)
    
        dst[:,outputStartY:dstHeight+outputStartY,outputStartX:dstWidth+outputStartX]=databuf
        
#        dst = np.moveaxis(dst,0,2)
        
#        im = Image.fromarray(dst)
#        im.save("t.jpg")
        
#        dst = np.concatenate((dst, dst), axis=2)
#        print(dst.shape)
        return dst
    
    def calculate_left_top_point_and_size(self,midPointX, midPointY, windowLength):
        leftTopX = int(midPointX - windowLength/2)
        leftTopY =int( midPointY - windowLength/2)
        listResult = (int(leftTopX),int(leftTopY),int(windowLength),int(windowLength))
        return listResult

def test():        
    train_dataset = RemoteSensingDataset(configs.image_path, configs.points_path, is_train = True, train_num = 1000,polygon_path = None)
    train_loader = DataLoader( train_dataset,  batch_size = 1, shuffle=False)   
    
    for i, data in enumerate(train_loader):
        positive_label, anchor_label, negative_label, pos_inner, pos_obj, pos_scene, anc_inner, anc_obj, anc_scene , neg_inner, neg_obj, neg_scene = data    
        print(pos_inner.size(), pos_obj.size(), pos_scene.size(),
              anc_inner.size(), anc_obj.size(), anc_scene.size(),
              neg_inner.size(), neg_obj.size(), neg_scene.size()
              )    
        
    # test_dataset = RemoteSensingDataset(configs.image_path, configs.points_path, is_train = False, train_num = 0 ,polygon_path = configs.test_polygon_path)
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)   
    # for i, data in enumerate(test_loader):
    #     object_data, neighbor_datas = data
    #     print(object_data.size(), neighbor_datas.size())
        
    #     object_data = torch.squeeze(object_data, dim=0)
    #     neighbor_datas =  torch.squeeze(neighbor_datas, dim=0)
        
    #     print(object_data.size(), neighbor_datas.size())
    #     if i == 0:
            
    #         break
#        data_inner, data_obj, data_sense = data
#        print(i,data_inner.size(), data_obj.size(), data_sense.size())
        
        
if __name__ == "__main__":
    test()
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    