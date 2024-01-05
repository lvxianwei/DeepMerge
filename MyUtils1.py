# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:04:43 2021

@author: lxw
"""

from torch.utils.data import Dataset, DataLoader
import torch
from osgeo import gdal
from osgeo import ogr
import os
import numpy as np
import cv2
from config import configs
import random
 
class MergingSegmensPairDataset(Dataset):
    
    def __init__(self, image_folder, polygon_folder,point_folder, positive_folder, negative_folder, num = 0):
        self.image_folder = image_folder
        self.polygon_folder = polygon_folder
        self.point_folder = point_folder
        self.num = num
        self.positive_folder = positive_folder
        self.negative_folder = negative_folder
        
        self.data = []
        self.point_dataset = []#为了不让系统清理dataset，需要暂存起来
        self.img_dataset = {}
        self.layers ={}
        positive_dataset = self.get_all_files(self.positive_folder)
        negative_dataset = self.get_all_files(self.negative_folder) 
        
        self.positive_number, self.positive_pair_number = self.add_data(positive_dataset, 1)
        self.negative_number,  self.negative_pair_number = self.add_data(negative_dataset, 0)
        print("OK")
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        layer = self.layers[item[0]]    
        left_pt_id = int(item[1])
        right_pt_id = int(item[2])
        flag = int(item[3])
        # left_pt_label = int(item[4])
        # right_pt_label = int(item[5])
        
        
        img_ds   = self.img_dataset[item[0]]
        
        left_point =layer.GetFeature(left_pt_id)
        left_meta_data = self.get_all_features(img_ds, left_point)
        right_point = layer.GetFeature(right_pt_id)
        right_meta_data = self.get_all_features(img_ds, right_point)
        
        return left_meta_data, right_meta_data, flag #, left_pt_label, right_pt_label
    
    def get_all_features(self, img_ds, feature):
        
        designed_features = self.get_designed_features(feature)
        designed_features = torch.Tensor(designed_features).unsqueeze(0)
        inner_scale  = int(feature.GetField("inner"))
        object_scale = int(feature.GetField("object"))
        scales, factors = self.get_scales(inner_scale, object_scale)  
        geometry = feature.GetGeometryRef()
        XGeo = geometry.GetX()
        YGeo = geometry.GetY()
        geoTrans = img_ds.GetGeoTransform()
        XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
        YLine  = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
        patches = self.get_patches_by_scales(img_ds, XPixel, YLine, scales)
        scales = torch.Tensor(scales).unsqueeze(0)
        factors = torch.Tensor(factors).unsqueeze(0)
        designed_features = torch.cat((designed_features, factors),dim=1)
        return designed_features, scales, patches
    
    def get_designed_features(self, feature):
        designed_features = []

        area = float(feature.GetField("area"))
        perimeter = float(feature.GetField("peri"))
        length = float(feature.GetField("len"))
        width = float(feature.GetField("width"))
        smoothness = float(feature.GetField("smooth"))
        std0 = float(feature.GetField("std0"))
        std1 = float(feature.GetField("std1"))
        std2 = float(feature.GetField("std2"))
        mean0 = float(feature.GetField("mean0"))
        mean1 = float(feature.GetField("mean1"))
        mean2 = float(feature.GetField("mean2"))
        shapeness = float(feature.GetField("shapeness"))
        compactness = float(feature.GetField("compact"))
        brightness = float(feature.GetField("bright"))
        borderindex = float(feature.GetField("border"))
        
        designed_features.append(area)
        designed_features.append(perimeter)
        designed_features.append(length)
        designed_features.append(width)
        designed_features.append(smoothness)
        designed_features.append(std0)
        designed_features.append(std1)
        designed_features.append(std2)
        designed_features.append(mean0)
        designed_features.append(mean1)
        designed_features.append(mean2)
        designed_features.append(shapeness)
        designed_features.append(compactness)
        designed_features.append(brightness)
        designed_features.append(borderindex) 
        
        return designed_features
    
    def get_patches_by_scales(self, img_ds, XPixel, YLine,  adaptive_scales):
        scale_count = len(adaptive_scales)
        regions = []
        for i in range(0, scale_count):
            adaptive_scale = adaptive_scales[i]
            # adaptive_scale = 224
            extract_data = self.calculate_left_top_point_and_size(XPixel, YLine, adaptive_scale)
            region = self.cut_image(img_ds, extract_data)
            region = self.resize_data(region, configs.scales[i], configs.scales[i])
            # region = self.resize_data(region,224, 224)
            regions.append(region)
        return regions
            
    
    def get_scales(self, inner_scale, object_scale):     
        interval = int(object_scale - inner_scale)
        pre_inter = interval
        # if interval <= 20:
        #     interval = int(interval * 0.4)
        
        # else:
        #     interval = int(interval * 0.4)
        #     object_scale = inner_scale + interval
        # print(pre_inter,interval)
        
        
        
        scene_scale  = object_scale + interval
        # scene_scale = object_scale + 5
        envi_scale  = object_scale + interval * 2
        # envi_scale = object_scale + 10
        scales = [inner_scale,  # inner_scale1,  inner_scale2,  inner_scale3,
                  object_scale, #object_scale1, object_scale2, object_scale3,
                  scene_scale,  #scene_scale1,  scene_scale2,  scene_scale3,
                  envi_scale]   #envi_scale1,   envi_scale2,   envi_scale3]
        factors = [inner_scale *1.0 / configs.scales[0],  # inner_scale1,  inner_scale2,  inner_scale3,
                  object_scale *1.0 / configs.scales[1], #object_scale1, object_scale2, object_scale3,
                  scene_scale *1.0 / configs.scales[2],  #scene_scale1,  scene_scale2,  scene_scale3,
                  envi_scale *1.0 / configs.scales[3]]
        # print(inner_scale,object_scale,scene_scale,envi_scale,"####",pre_inter,interval)
        return scales, factors
    
    
    
         
    
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
    
    def resize_data(self, data, target_high, target_width):
        data = data.transpose(1,2,0)
        data_img = np.zeros([target_high, target_width, 1])
        for i in range(0, self.band_num):
            data_band = data[:,:,i]
            data_band = cv2.resize(np.array(data_band), (target_high, target_width),interpolation=cv2.INTER_AREA)
            data_band = data_band[:,:, np.newaxis]
            
            if i == 0:
                data_img = data_band
            else:
                data_img = np.concatenate((data_img, data_band),axis = 2)
        data_img = data_img.transpose(2,0,1)
        data_img = data_img.astype(np.float32) / 255.0
        return data_img
               
    
    def calculate_left_top_point_and_size(self,midPointX, midPointY, windowLength):
        leftTopX = int(midPointX - windowLength/2)
        leftTopY = int(midPointY - windowLength/2)
        listResult = (int(leftTopX),int(leftTopY),int(windowLength),int(windowLength))
        return listResult
        
    def open_traing_dataset_txt(self, txt_path):
        result = []
        with open(txt_path,"r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(',')
                # pair = [line[1], line[2],line[3],line[4]]
                pair = [line[1], line[2]]
                result.append(pair)
        return result
                
    def add_data(self, txt_path, flag):
        if txt_path == None:
            return None
        count = 0
        parir_count = 0
        for i in range(0,len(txt_path)):
            train_file_name = txt_path[i].split('\\')[-1]
            train_file_name_without_extention = train_file_name.split('.')[0]
            polygon_file_name = "{0}.shp".format(train_file_name_without_extention)
            polygon_file_path = "{0}\\{1}".format(self.polygon_folder,polygon_file_name)
            point_file_name = "PointsGCS.shp"
            point_file_path = "{0}\\{1}\\{2}".format(self.point_folder,train_file_name_without_extention,point_file_name)
            image_file_name = "{0}.tif".format(train_file_name_without_extention)
            image_file_path = "{0}\\{1}".format(self.image_folder, image_file_name)
            _poly_ds, polygon_layer = self.open_vector_as_ds_and_layer(polygon_file_path)
            _pt_ds, point_layer   = self.open_vector_as_ds_and_layer(point_file_path)
            _img_ds = self.open_image_as_dataset(image_file_path)
            self.point_dataset.append(_pt_ds)
            self.img_dataset[train_file_name_without_extention] = _img_ds
            self.band_num = _img_ds.RasterCount
            self.layers[train_file_name_without_extention] = point_layer
            result = self.open_traing_dataset_txt(txt_path[i])
            # print(len(result))
            for j in range(0, len(result)):    
                left_polygon_id  = int(result[j][0])
                right_polygon_id = int(result[j][1])
                
                # left_polygon_label = int(result[j][2])
                # right_polygon_label = int(result[j][3])
                
                left_polygon  = polygon_layer.GetFeature(left_polygon_id)
                right_polygon = polygon_layer.GetFeature(right_polygon_id)
                left_poly_samples = left_polygon.GetField("PointID")
                right_poly_samples= right_polygon.GetField("PointID")
                
                left_poly_samples = left_poly_samples.split(' ')
                right_poly_samples= right_poly_samples.split(' ')
                
                #修改这里
                for m in range(0, len(left_poly_samples)):
                    for n in range(0, len(right_poly_samples)):
                        
                        m_rand = random.randint(0, len(left_poly_samples) - 1)
                        n_rand = random.randint(0, len(right_poly_samples) - 1)

                        left_sample  = left_poly_samples[m_rand]
                        right_sample = right_poly_samples[n_rand]
                        
                        
                        # left_sample  = left_poly_samples[m]
                        # right_sample = right_poly_samples[n]
                        # item = [train_file_name_without_extention,left_sample,right_sample,flag,left_polygon_label, right_polygon_label]
                        item = [train_file_name_without_extention,left_sample,right_sample,flag]

                        self.data.append(item)
                        count = count + 1
                        break
                    break
            parir_count = parir_count + len(result)
        return count, parir_count
        
        
    def get_all_files(self,cwd):
        if cwd == "":
            return None
        get_dir = os.listdir(cwd)
        result = []
        for i in get_dir:
            sub_dir = os.path.join(cwd, i)
            # print(sub_dir)
            result.append(sub_dir)
        return result
    
    def open_vector_as_ds_and_layer(self, shapefile_path):
        vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
        vector_DataSource = vector_deriver.Open(shapefile_path,0)  
        if vector_DataSource is None:
            raise ValueError("Can not open {0}".format(shapefile_path))
        layer = vector_DataSource.GetLayer(0)
        if layer is None:
            raise ValueError("Can not open {0}".format(shapefile_path))        
        return vector_DataSource, layer
    
    def open_image_as_dataset(self, image_path):
        image_ds = gdal.Open(image_path,gdal.GA_ReadOnly)
        if image_ds == None:
            raise ValueError("Can not open {0}".format(image_path))
        return image_ds
        
        
if __name__ == "__main__":
    image_folder    = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images"
    polygon_folder  = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    point_folder    = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    positive_folder = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\PositiveData"
    negative_folder = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\NegativeData"
    test_folder = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\TestData"
    data_set = MergingSegmensPairDataset(image_folder, polygon_folder,point_folder,  positive_folder, negative_folder)
    train_loader = DataLoader( data_set,  batch_size = 10, shuffle = True )
    for i, data in enumerate(train_loader):
        left_meta_data, right_meta_data, flag = data
        designed_features, scales, patches = left_meta_data
        # print("area = {0}".format(designed_features[0]))
        # print("perimeter = {0}".format(designed_features[1]))
        # print("length = {0}".format(designed_features[2]))
        # print("width = {0}".format(designed_features[3]))
        # print("smoothness = {0}".format(designed_features[4]))
        # print("std1 = {0}".format(designed_features[5]))
        # print("std2 = {0}".format(designed_features[6]))
        # print("std3 = {0}".format(designed_features[7]))
        # print("mean0 = {0}".format(designed_features[8]))
        # print("mean1 = {0}".format(designed_features[9]))
        # print("mean2 = {0}".format(designed_features[10]))
        # print("shapeness = {0}".format(designed_features[11]))
        # print("compactness = {0}".format(designed_features[12]))
        # print("brightness = {0}".format(designed_features[13]))
        # print("borderindex = {0}".format(designed_features[14]))
        # print(designed_features.shape)
        # for j in range(0, 4):
        # print(i)
        print(designed_features.shape)
        # designed_features = torch.cat((designed_features, factors),dim = 2)
        # print(designed_features.shape)
        print(scales)

        # print(factors)
        break
    print(len(data_set))
    print("positive number: ", data_set.positive_pair_number, data_set.positive_number)
    print("negative number: ", data_set.negative_pair_number, data_set.negative_number)
    
    
    
    
    
    
    
    
    