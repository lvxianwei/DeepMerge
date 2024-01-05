# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 21:23:14 2022

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


class PolygonPointPairDataset(Dataset):
    
    def __init__(self, image_folder, polygon_folder,point_folder, num = 0):
        self.image_folder = image_folder
        self.polygon_folder = polygon_folder
        self.point_folder = point_folder
        self.num = num
        
        self.data = []
        self.polygon_dataset = []
        self.point_dataset = []#为了不让系统清理dataset，需要暂存起来
        self.line_dataset = []
        self.img_dataset = {}
        self.polygon_layers = {}
        self.point_layers ={}
        self.line_layers = {}
        test_files = self.get_all_files(self.polygon_folder, ".shp")
        
        count = self.add_data(test_files)
        print("OK ",count)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
                
    def add_data(self, txt_path):
        if txt_path == None:
            return None
        count = 0
        for i in range(0,len(txt_path)):
            # i = 0
            i = 17#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            train_file_name = txt_path[i].split('\\')[-1]
            print(train_file_name)
            train_file_name_without_extention = train_file_name.split('.')[0]
            polygon_file_name = "{0}.shp".format(train_file_name_without_extention)
            polygon_file_path = "{0}\\{1}".format(self.polygon_folder,polygon_file_name)
            point_file_name = "PointsGCS.shp"
            point_file_path = "{0}\\{1}\\{2}".format(self.point_folder,train_file_name_without_extention,point_file_name)
            
            line_file_name = "lines.shp"
            line_file_path = "{0}\\{1}\\{2}".format(self.point_folder,train_file_name_without_extention,line_file_name)
            image_file_name = "{0}.tif".format(train_file_name_without_extention)
            image_file_path = "{0}\\{1}".format(self.image_folder, image_file_name)
            
            _poly_ds, polygon_layer = self.open_vector_as_ds_and_layer(polygon_file_path)
            _pt_ds, point_layer   = self.open_vector_as_ds_and_layer(point_file_path)
            _line_ds, line_layer = self.open_vector_as_ds_and_layer(line_file_path)
            
            _img_ds = self.open_image_as_dataset(image_file_path)
            
            self.polygon_dataset.append(_poly_ds)
            self.point_dataset.append(_pt_ds)
            self.line_dataset.append(_line_ds)
            self.img_dataset[train_file_name_without_extention] = _img_ds
            self.band_num = _img_ds.RasterCount
            self.point_layers[train_file_name_without_extention] = point_layer
            self.polygon_layers[train_file_name_without_extention] = polygon_layer
            self.line_layers[train_file_name_without_extention] = line_layer
            line_layer.ResetReading()
            feature = line_layer.GetNextFeature()
            while feature is not None:
                left_polygon_id  = int(feature.GetField("LEFT_FID"))
                right_polygon_id = int(feature.GetField("RIGHT_FID"))
                feature_id       = int(feature.GetFID())
                # print(train_file_name_without_extention, left_polygon_id, right_polygon_id)
                if left_polygon_id == -1 or right_polygon_id == -1:
                    feature = line_layer.GetNextFeature()
                    continue
                else:
                    
                    pair = [feature_id,train_file_name_without_extention,left_polygon_id, right_polygon_id]
                    self.data.append(pair)
                    count = count + 1
                    feature = line_layer.GetNextFeature()
            break
        return count
        
        
    def get_all_files(self,cwd, file_type):
        if cwd == "":
            return None
        get_dir = os.listdir(cwd)
        result = []
        for i in get_dir:
            sub_dir = os.path.join(cwd, i)
           
            if sub_dir.endswith(file_type):
                # print(sub_dir)
                result.append(sub_dir)
        return result
    
    def open_vector_as_ds_and_layer(self, shapefile_path):
        vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
        vector_DataSource = vector_deriver.Open(shapefile_path,1)  
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
    
class PolygonConnectPointDataset(Dataset):
    
    def __init__(self, image_path, polygon_path, polyline_path, point_path, num = 0):
        self.image_path = image_path
        self.polygon_path = polygon_path
        self.polyline_path = polyline_path
        self.point_path = point_path
        self.num = num
        
        self.data = []
        self.polygon_dataset = None
        self.point_dataset = None#为了不让系统清理dataset，需要暂存起来
        self.line_dataset = None
        self.img_dataset = None
        self.polygon_layer = None
        self.point_layer =None
        self.line_layer = None
        
        count = self.add_data(self.polygon_path)
        print("OK ",count)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
                
    def add_data(self, txt_path):
        if txt_path == None:
            return None
        count = 0
        train_file_name = txt_path.split('\\')[-1]
        print(train_file_name)
        train_file_name_without_extention = train_file_name.split('.')[0]
            
        _poly_ds, polygon_layer = self.open_vector_as_ds_and_layer( self.polygon_path)
        _pt_ds, point_layer   = self.open_vector_as_ds_and_layer(self.point_path)
        _line_ds, line_layer = self.open_vector_as_ds_and_layer(self.polyline_path)
        
        _img_ds = self.open_image_as_dataset(self.image_path)
        
        self.polygon_dataset = _poly_ds
        self.point_dataset = _pt_ds
        self.line_dataset = _line_ds
        self.img_dataset = _img_ds
        self.band_num = _img_ds.RasterCount
        self.point_layer = point_layer
        self.polygon_layer = polygon_layer
        self.line_layer = line_layer
        self.line_layer.ResetReading()
        feature = self.line_layer.GetNextFeature()
        while feature is not None:
            left_polygon_id  = int(feature.GetField("LEFT_FID"))
            right_polygon_id = int(feature.GetField("RIGHT_FID"))
            feature_id       = int(feature.GetFID())
            # print(train_file_name_without_extention, left_polygon_id, right_polygon_id)
            if left_polygon_id == -1 or right_polygon_id == -1:
                feature = self.line_layer.GetNextFeature()
                continue
            else:
                
                pair = [feature_id,train_file_name_without_extention,left_polygon_id, right_polygon_id]
                self.data.append(pair)
                count = count + 1
                feature = self.line_layer.GetNextFeature()
        return count
    
    def open_vector_as_ds_and_layer(self, shapefile_path):
        vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
        vector_DataSource = vector_deriver.Open(shapefile_path,1)  
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

    

class ExtractFeatureDataset(Dataset):

    def __init__(self, image_path ,point_path):
        self.image_path = image_path
        self.point_path = point_path
        self.data = []
        count = self.add_data(image_path,point_path)
        # print("OK ",count)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx = self.data[index]
        point =self.point_layers.GetFeature(idx)
        meta_data = self.get_all_features(self.img_dataset, point)
        return idx, meta_data
    
    def get_all_features(self, img_ds, feature):
        designed_features = self.get_designed_features(feature)
        designed_features = torch.Tensor(designed_features).unsqueeze(0)
        inner_scale  = int(feature.GetField("inner"))
        object_scale = int(feature.GetField("object"))
        scales,factors = self.get_scales(inner_scale, object_scale)  
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
    
    def get_patches_by_scales(self, img_ds, XPixel, YLine,  scales):
        scale_count = len(scales)
        regions = []
        for i in range(0, scale_count):
            scale = scales[i]
            # scale = 224
            extract_data = self.calculate_left_top_point_and_size(XPixel, YLine, scale)
            region = self.cut_image(img_ds, extract_data)
            region = self.resize_data(region, configs.scales[i], configs.scales[i])
            # region = self.resize_data(region, 224, 224)
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
        
        
        
        scene_scale  = object_scale + interval * 1  
        envi_scale  = object_scale + interval * 2
        # scene_scale  = object_scale + 5
        # envi_scale  = object_scale + 10
        scales = [inner_scale, 
                  object_scale, 
                  scene_scale, 
                  envi_scale]
        
        factors = [inner_scale *1.0 / configs.scales[0],  # inner_scale1,  inner_scale2,  inner_scale3,
                  object_scale *1.0 / configs.scales[1], #object_scale1, object_scale2, object_scale3,
                  scene_scale *1.0 / configs.scales[2],  #scene_scale1,  scene_scale2,  scene_scale3,
                  envi_scale *1.0 / configs.scales[3]]
        
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
        leftTopY =int( midPointY - windowLength/2)
        listResult = (int(leftTopX),int(leftTopY),int(windowLength),int(windowLength))
        return listResult
                
    def add_data(self, img_path, point_path):
        if point_path == None or img_path == None:
            return None
        count = 0

        # train_file_name = point_path.split('\\')[-1]
        # print(train_file_name)
        _pt_ds, point_layer   = self.open_vector_as_ds_and_layer(point_path)
        _img_ds = self.open_image_as_dataset(img_path)
            
        self.point_dataset = _pt_ds
        self.img_dataset = _img_ds
        self.band_num = _img_ds.RasterCount
        self.point_layers = point_layer
        
        point_layer.ResetReading()
        feature = point_layer.GetNextFeature()
        while feature is not None:
            feature_id = int(feature.GetFID())
            self.data.append(feature_id)
            count = count + 1
            feature = point_layer.GetNextFeature()
        return count
        
        
    def get_all_files(self,cwd, file_type):
        if cwd == "":
            return None
        get_dir = os.listdir(cwd)
        result = []
        for i in get_dir:
            sub_dir = os.path.join(cwd, i)
           
            if sub_dir.endswith(file_type):
                # print(sub_dir)
                result.append(sub_dir)
        return result
    
    def open_vector_as_ds_and_layer(self, shapefile_path):
        vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
        vector_DataSource = vector_deriver.Open(shapefile_path,1)  
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
    image_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images\PhoenixCityGroup05_05_2.tif"
    point_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro\PhoenixCityGroup05_05_2\PointsGCS.shp"
    data_set = ExtractFeatureDataset(image_path, point_path)
    test_loader = DataLoader( data_set,  batch_size = 10, shuffle = False )
    for i, data in enumerate(test_loader):
        idx, left_meta_data = data
        designed_features, scales, patches = left_meta_data
        # print(designed_features.shape)
        # print("area = {0}".format(designed_features[0][0][0]))
        # print("perimeter = {0}".format(designed_features[0][0][1]))
        # print("length = {0}".format(designed_features[0][0][2]))
        # print("width = {0}".format(designed_features[0][0][3]))
        # print("smoothness = {0}".format(designed_features[0][0][4]))
        # print("std1 = {0}".format(designed_features[0][0][5]))
        # print("std2 = {0}".format(designed_features[0][0][6]))
        # print("std3 = {0}".format(designed_features[0][0][7]))
        # print("mean0 = {0}".format(designed_features[0][0][8]))
        # print("mean1 = {0}".format(designed_features[0][0][9]))
        # print("mean2 = {0}".format(designed_features[0][0][10]))
        # print("shapeness = {0}".format(designed_features[0][0][11]))
        # print("compactness = {0}".format(designed_features[0][0][12]))
        # print("brightness = {0}".format(designed_features[0][0][13]))
        # print("borderindex = {0}".format(designed_features[0][0][14]))
        
        # for j in range(0, 4):
        #     print(i,scales[j], patches[j].shape)
        break
    print(len(data_set))
    