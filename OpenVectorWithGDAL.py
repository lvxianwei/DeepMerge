#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:47:29 2018

@author: lxw
"""

# import tensorflow as tf
import os
import numpy as np
import random
#import math
from osgeo import gdal
#from osgeo.gdalconst import *
from osgeo import ogr
from PIL import Image


gdal.AllRegister()

#*********************************************************************************************************************************************
#*********************************************************************************************************************************************
#*********************************************************************************************************************************************
#the parameters set by user, please set parameters
#test_name = "CenterID.shp"   
#test_name = "MBBForClassification.shp"
#test_name = "RandomOverall.shp"
#test_name = "RandomPerObject.shp"
test_name = "01011.shp"
#test_name = "ConcreteRoadPoints.shp"
#test_name = "ZhhongYanfei_Moon.shp" 
#test_name = "RandomOveralConnectZhouWenl.shp"  
#test_name = "ResidencePoints.shp"
#test_name = "FactoryExamplePoints.shp"    
#test_name = "FactoryPoints.shp" 
Test_data_name =test_name.split('.')[0]

train_name = "OverallTrainData15762.shp"                                                                     
image_name = "PhoenixCityGroup01_01_1.tif"#image
#zone_name = "Data"
data_size = 250 #输入图像大小
data_ratio = 1 #提取数据占标记数据比例
train_ratio = 0.625 #训练数据占提取数据比例
num_train_samples = 15762 #Overall 
classes = ['AsphaltRoad', 'BareSoil', 'ConcreteRoad', 'Container', 'IndustryBuilding', 'Residence','Shadow','Vegetation','Water','WetLand'] #moon water 数据标签
IsWriteTrain = False
IsWriteVsl = False
IsWriteTest = True
#*********************************************************************************************************************************************
#*********************************************************************************************************************************************
#*********************************************************************************************************************************************

cwd = os.getcwd() #当前代码路径
test_path = "{0}/{1}/{2}/{3}".format(cwd,"Data","Test-Data",test_name)

train_path = "{0}/{1}/{2}/{3}".format(cwd,"Data","Labeled-Data",train_name)
image_path = "{0}/{1}/{2}/{3}".format(cwd,"Data","Images",image_name)

def IsExistFolderAndCreateFolder(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)
        print("Create folder path:",folder)
    else:
        print("Folder {0} exists!".format(folder))
data_file_folder = "{0}/{1}/Tf-Records".format(cwd,"Data")
IsExistFolderAndCreateFolder(data_file_folder)
train_file_path = "{0}/train-{1}.tfrecords".format(data_file_folder,data_size)
val_file_path = "{0}/val-{1}.tfrecords".format(data_file_folder,data_size)
test_file_folder = "{0}/{1}".format(data_file_folder,Test_data_name)
IsExistFolderAndCreateFolder(test_file_folder)
test_file_path = "{0}/test-{1}.tfrecords".format(test_file_folder,data_size) 
intervalFlag = int(num_train_samples * data_ratio * train_ratio + 0.5)
#print intervalFlag
temp_dictory = []
if IsWriteTrain == True or IsWriteVsl == True:
    ds = gdal.Open(image_path,gdal.GA_ReadOnly)
    vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
    vector_DataSource = vector_deriver.Open(train_path,0)  
    if vector_DataSource is None:
        print ("could not open!") 
    layer = vector_DataSource.GetLayer(0)
    num_features = layer.GetFeatureCount()
    for num in range(0,num_features):
        temp_dictory.append(num)
    random.seed(10) #设定随机种子
    random.shuffle(temp_dictory) #随机打乱图像列表  

#calculate the points start and end of region image from orginal image
def CalculateStartAndEndPoints(midPointX, midPointY, windowLength):
    leftTopX = int(midPointX - windowLength/2)
    leftTopY =int( midPointY - windowLength/2)
#    rigthBottomX = leftTopX +windowLength
#    rigthBottomY = leftTopY + windowLength
    listResult = (int(leftTopX),int(leftTopY),int(windowLength),int(windowLength))
    return listResult

def CutImage(ds,startOrgX,startOrgY,dstWidth,dstHeight):
    bandsNum = ds.RasterCount
    srcWidth = ds.RasterXSize#源图像宽
    srcHeight=ds.RasterYSize
#    print(bandsNum,srcWidth,srcHeight)
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
    databuf = ds.ReadAsArray(int(startX),int(startY),int(dstWidth), int(dstHeight))
    dst=np.zeros((bandsNum,outputHeight,outputWidth),dtype=np.uint8)
    
    dst[:,outputStartY:dstHeight+outputStartY,outputStartX:dstWidth+outputStartX]=databuf
    dst = np.moveaxis(dst,0,2)
#    im = Image.fromarray(dst)
#    im.save("t.jpg")
    return dst

# def _int64_feature(value):
    # return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

# def _bytes_feature(value):
    # return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def WriteTest(flag = True):
    if flag == False:
        print ("Didn't WriteTest!")
        return    
    ds = gdal.Open(image_path,gdal.GA_ReadOnly)
    geoTrans = ds.GetGeoTransform()
#    img = Image.open(image_path)
    #vector data processing
    vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
    vector_DataSource = vector_deriver.Open(test_path,0)
    if vector_DataSource is None:
        print ("could not open!") 
    # writer = tf.python_io.TFRecordWriter(test_file_path)
    layer = vector_DataSource.GetLayer(0)
    i = 0
    
    while True:
        feature = layer.GetNextFeature()   
        if feature is None:
            break
        else:
            
            geometry = feature.GetGeometryRef()
            print("pt: ", i)
            XGeo = geometry.GetX()
            YGeo = geometry.GetY()
            # print(XGeo, YGeo)
            # print(geoTrans)
            XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
            YLine = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
            print("X ",XPixel,"Y: ", YLine )
            lstPoints = CalculateStartAndEndPoints(XPixel,YLine,data_size)
            region =  CutImage(ds,lstPoints[0],lstPoints[1],lstPoints[2],lstPoints[3])
            im = Image.fromarray(region)
            im.save("{0}\\{1}_{2}_{3}.jpg".format(test_file_folder,i+1, XPixel,YLine))
            fid = feature.GetFID()
            formation = "Feature: {0}, {1}, {2}.".format(fid,XPixel,YLine)
            print (formation)
            i = i + 1
            if i ==1000:
                break
    # writer.close()
    layer.ResetReading()
    
def WriteTestPerPixel(flag = True):
    if flag == False:
        print ("Didn't WriteTest!")
        return    
    img = Image.open(image_path)
    width,height = img.size
    # writer = tf.python_io.TFRecordWriter(test_file_path)
    for XPixel in range(0,width):
        for YLine in range(0,height):
            lstPoints = CalculateStartAndEndPoints(XPixel,YLine,data_size)         
            region = img.crop(lstPoints)
            img_raw = region.tobytes()
            # example = tf.train.Example(features = tf.train.Features(feature = {'img_raw': _bytes_feature(img_raw),                                                                         'img_name': _bytes_feature(str(XPixel) +'_' +str(YLine))}))
            # writer.write(example.SerializeToString())
            formation = "Feature: {0}, {1}, {2}.".format(XPixel,'_',YLine)
            print (formation)  
    # writer.close()

def WriteTrain(flag = True):
    if flag == False:
        print ("Didn't WriteTrain!")
        return
    
    ds = gdal.Open(image_path,gdal.GA_ReadOnly)
#    CutImage(ds,(-30,-10,150,200))
    geoTrans = ds.GetGeoTransform()
#    print('gdal run correctly')
#    img = Image.open(image_path)
    #vector data processing
    vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
    vector_DataSource = vector_deriver.Open(train_path,0)
    
    if vector_DataSource is None:
        print ("could not open!") 
    # writer = tf.python_io.TFRecordWriter(train_file_path)
    layer = vector_DataSource.GetLayer(0)
#    dirctory = {}
#    for index, name in enumerate(classes):# initialize the directory by empty list
#        dirctory[name] = []
        
#    while True:#initialize the directory by the labeled data
#        feature = layer.GetNextFeature()   
#        if feature is None:
#            break
#        else:
#            label = feature.GetField('Class_name')
#            index = feature.GetFID()
#            dirctory[label].append(index)
#            print index,label
#    layer.ResetReading()   

#    for index, name in enumerate(classes):
#        print ("Label: %s; Class name: %s" % (index, name))
#        
#        dirctory[name].sort()
#        random.seed(10) #设定随机种子
#        random.shuffle( dirctory[name]) #随机打乱图像列表
#        train_list = dirctory[name][0:int(math.ceil(len(dirctory[name]) * train_ratio * data_ratio))] #name类训练图像文件名列表
#
#        for t in range(len(train_list)):
#            feature = layer.GetFeature(train_list[t])  
#            if feature is None:
#                return  
#            else:         
#                geometry = feature.GetGeometryRef()
#                XGeo = geometry.GetX()
#                YGeo = geometry.GetY()
#                XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
#                YLine = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
#                lstPoints = CalculateStartAndEndPoints(XPixel,YLine,data_size)
#                train_image = CutImage(ds,lstPoints)       
#                train_name = "{0}_{1}".format(XPixel,YLine)
##                print train_name
#                train_image = train_image.resize((data_size, data_size)) #Resize图像
#                img_raw = train_image.tobytes() #将图像转化为byte格式
#                example = tf.train.Example(features = tf.train.Features(feature = {'label': _int64_feature(index),
#                                                                                   'img_raw': _bytes_feature(img_raw),
#                                                                                   'img_name': _bytes_feature(train_name)}))
#                writer.write(example.SerializeToString())
#                print index, train_name
    
    
#modified by lv xianwei in 2019.07.18

    
    
    my_dictory = temp_dictory[0:intervalFlag]
    cate_dictory = {'AsphaltRoad':0, 'BareSoil':0, 'ConcreteRoad':0, 'Container':0, 'IndustryBuilding':0
                    , 'Residence':0,'Shadow':0,'Vegetation':0,'Water':0,'WetLand':0} #moon water 数据标签
#    print my_dictory.count
#    print num_features 
#    qqq = 0
    for i in my_dictory:
#        qqq+=1
#        print qqq
#        if flag == 
        feature = layer.GetFeature(i)
        label = feature.GetField('Class_name')
        index = 0
        for j, name in enumerate(classes):
            if name == label:
                index = j     
                cate_dictory[name] = cate_dictory[name] + 1
        geometry = feature.GetGeometryRef()
        XGeo = geometry.GetX()
        YGeo = geometry.GetY()
        
        XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
        YLine = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
        lstPoints = CalculateStartAndEndPoints(XPixel,YLine,data_size)
        train_image = CutImage(ds,lstPoints[0],lstPoints[1],lstPoints[2],lstPoints[3])       
        train_name = "{0}_{1}".format(XPixel,YLine)
        
#        save_path = "{0}\\PICS\\{1}".format(data_file_folder,label)
#        IsExistFolderAndCreateFolder(save_path)
#        im = Image.fromarray(train_image)
#        im.save("{0}\\{1}.jpg".format(save_path,train_name))
        
        
        train_image = train_image.resize((data_size, data_size)) #Resize图像
        img_raw = train_image.tobytes() #将图像转化为byte格式
        # example = tf.train.Example(features = tf.train.Features(feature = {'label': _int64_feature(index),
        #                                                                            'img_raw': _bytes_feature(img_raw),
        #                                                                            'img_name': _bytes_feature(train_name.encode())}))
        # writer.write(example.SerializeToString())
#        print (index,train_name,label)     
#        print index,label  
    # writer.close()
    total = 0
    for value in cate_dictory.values():
        total = value + total
    print("train",cate_dictory,total, len(my_dictory))      
    
            
def WriteVal(flag = True):
    if flag == False:
        print ("Didn't WriteVal!")
        return
    ds = gdal.Open(image_path,gdal.GA_ReadOnly)
    geoTrans = ds.GetGeoTransform()
#    img = Image.open(image_path)
    #vector data processing
    vector_deriver = ogr.GetDriverByName('ESRI Shapefile')
    vector_DataSource = vector_deriver.Open(train_path,0)
    if vector_DataSource is None:
        print ("could not open!") 
    # writer = tf.python_io.TFRecordWriter(val_file_path)
    layer = vector_DataSource.GetLayer(0)
#    dirctory = {}
#    for index, name in enumerate(classes):# initialize the directory by empty list
#        dirctory[name] = []
#        
#    while True:#initialize the directory by the labeled data
#        feature = layer.GetNextFeature()   
#        if feature is None:
#            break
#        else:
#            label = feature.GetField('Class_name')
#            index = feature.GetFID()
#            dirctory[label].append(index)
##            print index,label
#    layer.ResetReading()   
#
#    for index, name in enumerate(classes):
#        print ("Label: %s; Class name: %s" % (index, name))
#        
#        dirctory[name].sort()
#        random.seed(10) #设定随机种子
#        random.shuffle( dirctory[name]) #随机打乱图像列表
#        val_list = dirctory[name][int(math.ceil(len(dirctory[name]) * train_ratio * data_ratio)):int(math.ceil(len(dirctory[name]) * data_ratio))] #name类训练图像文件名列表
#
#        for t in range(len(val_list)):
#            feature = layer.GetFeature(val_list[t])  
#            if feature is None:
#                return  
#            else:                
#                geometry = feature.GetGeometryRef()
#                XGeo = geometry.GetX()
#                YGeo = geometry.GetY()
#                XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
#                YLine = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
#                lstPoints = CalculateStartAndEndPoints(XPixel,YLine,data_size)
#                train_image = CutImage(ds,lstPoints)       
#                train_name = "{0}_{1}".format(XPixel,YLine)
#                train_image = train_image.resize((data_size, data_size)) #Resize图像
#                img_raw = train_image.tobytes() #将图像转化为byte格式
#                example = tf.train.Example(features = tf.train.Features(feature = {'label': _int64_feature(index),
#                                                                                   'img_raw': _bytes_feature(img_raw),
#                                                                                   'img_name': _bytes_feature(train_name)}))
#                writer.write(example.SerializeToString())
#                print index, train_name
    
    my_dictory = temp_dictory[intervalFlag:]
    cate_dictory = {'AsphaltRoad':0, 'BareSoil':0, 'ConcreteRoad':0, 'Container':0, 'IndustryBuilding':0
                    , 'Residence':0,'Shadow':0,'Vegetation':0,'Water':0,'WetLand':0} #moon water 数据标签
#    print intervalFlag, my_dictory.count
#    print num_features 
#    tttt = 0
    for i in my_dictory:
#        tttt += 1
#        print tttt
#        if flag == 
        feature = layer.GetFeature(i)
        label = feature.GetField('Class_name')
        index = 0
        for j, name in enumerate(classes):
            if name == label:
                index = j          
                cate_dictory[name] = cate_dictory[name] + 1
        geometry = feature.GetGeometryRef()
        XGeo = geometry.GetX()
        YGeo = geometry.GetY()
        XPixel = int(abs((geoTrans[0] - XGeo) / geoTrans[1]) + 1)
        YLine = int(abs((geoTrans[3] - YGeo) / geoTrans[5]) + 1)
        lstPoints = CalculateStartAndEndPoints(XPixel,YLine,data_size)
        train_image = CutImage(ds,lstPoints[0],lstPoints[1],lstPoints[2],lstPoints[3])       
        train_name = "{0}_{1}".format(XPixel,YLine)
            
#        train_image = train_image.resize((data_size, data_size)) #Resize图像
        img_raw = train_image.tobytes() #将图像转化为byte格式
#         example = tf.train.Example(features = tf.train.Features(feature = {'label': _int64_feature(index),
#                                                                                    'img_raw': _bytes_feature(img_raw),
#                                                                                    'img_name': _bytes_feature(train_name.encode())}))
#         writer.write(example.SerializeToString())
# #        print (index,train_name,label)     
#     writer.close()
    total = 0
    for value in cate_dictory.values():
        total = value + total
    print("validate",total,cate_dictory)

def main(argv = None):
    print("######################################################")
    WriteTest(IsWriteTest)
    print("######################################################")
    WriteTrain(IsWriteTrain)
    print("######################################################")
    WriteVal(IsWriteVsl)
    print ("Finished!")
if __name__ == '__main__':
    main()









































