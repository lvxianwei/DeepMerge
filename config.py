# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:32:59 2021

@author: lxw
"""


class DefaultConfigs(object):
    
    # points_path = r"F:\CalculatingScales\DataTest\test\PointsConnect.shp"
    points_path = "./data/test/PointsConnect.shp"
    # image_path =  r"F:\CalculatingScales\DataTest\imgs\Sacramento_trans.tif"
    image_path = "./data/imgs/Sacramento_trans.tif"
    # test_polygon_path = r"F:\CalculatingScales\DataTest\test\LabeldDataConnect.shp"
    test_polygon_path = "./data/test/LabeldDataConnect.shp"
    target_img_size = 224
    channel_num = 3
    batch_size = 120 #120 for 642
    image_statistic_path = "./data/imgs/image_statistic.txt"
    base_net = "VGG16"
    pooling = "GAP"
    out_channels = 512
    num_epochs = 100
    train_num = 1000
    result_dir = r"F:\SpatialSeriesTripletGUR"
    model_paras_path = "./model/"
    result_path = "./result/"
    traing_dataset_txt_path = "./data/"
    checkpoint_path = "./model/S2Former_v3-3CH-3DP-SEF-642_100epochs.pth"
    # scales = [28, 56, 112, 224]
    scales = [32, 64, 128, 1]
    # scales = [224, 224, 224, 224]
configs = DefaultConfigs()