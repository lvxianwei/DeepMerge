

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 21:00:40 2022

@author: lxw
"""
import torch
import h5py
from MyUtils2 import PolygonConnectPointDataset, ExtractFeatureDataset
from torch.utils.data import DataLoader
from config import configs
from vit_model import vit_base_patch_scales_224_in21k as create_model
import numpy as np
import os
from osgeo import ogr
import torch.nn.functional as F
from tqdm import tqdm
from nets.ShfitScaleFormer import ShfitScaleFormer, ShfitScaleFormer_v2, ShfitScaleFormer_v3,ShfitScaleFormer_v4,ShfitScaleFormer_v5,ShfitScaleFormer_v6


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class FeatureIO():
    def __init__(self, net, checkpoint_path):
        self.net = net
        self.checkpoint_path = checkpoint_path
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

        model_CKPT = torch.load(checkpoint_path)
        self.net.load_state_dict(model_CKPT['net'])
        print("Network is uploaded into GPUs")
        for param in self.net.parameters():
            param.requires_grad = False
        self.h5_file_path = ""
        self.h5py_file = None
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    def extract_features(self, image_path, point_path, h5_file_path, batch_size=2000):

        self.h5_file_path = h5_file_path
        self.h5py_file = h5py.File(self.h5_file_path, 'w')
        data_set = ExtractFeatureDataset(image_path, point_path)
        test_loader = DataLoader(
            data_set, batch_size=batch_size, shuffle=False)
        # print(len(data_set))
        # print(len(test_loader))
        # raise("test")
        test_bar = tqdm(test_loader)
        name = image_path.split('\\')[-1]
        # print("Processing in img: {0}".format(name))
        for i, data in enumerate(test_bar):
            # with torch.no_grad():

            idx, meta_data = data
            designed_features, scales, patches = meta_data
            # if torch.cuda.is_available():
            designed_features = designed_features.cuda()
            patches0 = patches[0].cuda()
            patches1 = patches[1].cuda()
            patches2 = patches[2].cuda()
            # patches3 = patches[3].cuda()
            # patches = [patches0, patches1, patches2, patches3]
            patches = [patches0,patches1,patches2]
            # print(patches0.shape)
            # print(patches1.shape)
            # print(patches2.shape)
            # print(patches3.shape)
            # print(designed_features.shape)
            out_features = self.net(patches, designed_features)
            data = out_features.cpu().detach().numpy()
            # print(data.shape)
            self.save_h5(self.h5py_file, data, "dataset")
            # print("{0}/{1}th batch has been saved into disk".format(i+1,len(test_loader)))
            test_bar.desc = "{0}/{1}th batch is saved".format(
                i+1, len(test_loader))
            # test_bar.update(1)
        self.h5py_file.close()
        # print("{0} is proccessed".format(name))
        return len(data_set), len(test_loader)

    def save_h5(self, h5f, data, dataset_name="dataset"):
        shape_list = list(data.shape)
        if not h5f.__contains__(dataset_name):
            shape_list[0] = None  # 设置数组的第一个维度是0
            dataset = h5f.create_dataset(
                dataset_name, data=data, maxshape=tuple(shape_list), chunks=True)
            return
        else:
            dataset = h5f[dataset_name]
            len_old = dataset.shape[0]
            len_new = len_old+data.shape[0]
            shape_list[0] = len_new
            dataset.resize(tuple(shape_list))  # 修改数组的第一个维度
            dataset[len_old:len_new] = data  # 存入新的文件

    def ReadFeatures(self, h5_file_path):
        f = h5py.File(h5_file_path, 'r')
        dataset = f['dataset']
        self.h5py_file = f
        self.dataset = dataset

    def GetFeaturesByID(self, idx):
        if idx >= len(self.dataset):
            raise("index error!")
        return self.dataset[idx]

    def Close(self):
        if self.h5py_file != None:
            self.h5py_file.close()


def Euclidean_distance(X, Y):
    """
    输入：
    X n*p数组, p为特征维度
    Y m*p数组
    输出：
    D n*m距离矩阵
    """

    # x_mean = np.mean(X,axis=0)
    # y_mean = np.mean(Y,axis=0)
    # x_mean = x_mean[np.newaxis,:]
    # y_mean = y_mean[np.newaxis,:]
    # print(x_mean.shape, y_mean.shape)
    # print(X.shape, Y.shape)
    # X = np.mean(X,axis=0)
    # Y = np.mean(Y,axis=0)
    # X = X[np.newaxis,:]
    # Y = Y[np.newaxis,:]
    # print(X.shape, Y.shape)
    n = X.shape[0]
    m = Y.shape[0]
    X2 = np.sum(X ** 2, axis=1)
    Y2 = np.sum(Y ** 2, axis=1)
    D = np.tile(X2.reshape(n, 1), (1, m)) + \
        (np.tile(Y2.reshape(m, 1), (1, n))).T - 2 * np.dot(X, Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D


def test_for_shp(featureIO):
    image_folder = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images"
    polygon_folder = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    point_folder = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"

    image_path = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images\PhoenixCityGroup05_05_2.tif"
    polygon_path = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro\PhoenixCityGroup05_05_2.shp"
    polyline_path = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro\PhoenixCityGroup05_05_2\lines.shp"
    point_path = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro\PhoenixCityGroup05_05_2\PointsGCS.shp"

    data_set = PolygonConnectPointDataset(
        image_path, polygon_path, polyline_path, point_path)
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False)
    print(len(test_loader))
    for i, data in enumerate(test_loader):
        fid,  name, left_polygon_id, right_polygon_id = data
        name = name[0]
        # similarity_txt = open(configs.result_path + name +".txt", 'w')
        # print(name)
        left_polygon_id = left_polygon_id[0]
        right_polygon_id = right_polygon_id[0]
        # print(name, left_polygon_id,right_polygon_id)
        layer = data_set.polygon_layer
        left_polygon = layer.GetFeature(left_polygon_id)
        right_polygon = layer.GetFeature(right_polygon_id)
        left_poly_samples = left_polygon.GetField("PointID")
        right_poly_samples = right_polygon.GetField("PointID")

        left_poly_samples = left_poly_samples.split(' ')
        right_poly_samples = right_poly_samples.split(' ')

        line_layer = data_set.line_layer
        defn = line_layer.GetLayerDefn()
        fieldIndex = defn.GetFieldIndex("simi")
        if fieldIndex < 0:
            fieldDefn = ogr.FieldDefn('simi', ogr.OFTReal)
            line_layer.CreateField(fieldDefn, 1)

        out_left_data = []
        out_right_data = []
        for m in range(0, len(left_poly_samples)):
            left_pt_id = int(left_poly_samples[m])
            left_features = featureIO.GetFeaturesByID(left_pt_id)
            left_features = left_features[np.newaxis, :]
            if m == 0:
                out_left_data = left_features
            else:
                out_left_data = np.concatenate(
                    (out_left_data, left_features), axis=0)
        for n in range(0, len(right_poly_samples)):
            right_pt_id = int(right_poly_samples[n])
            right_features = featureIO.GetFeaturesByID(right_pt_id)
            right_features = right_features[np.newaxis, :]
            if n == 0:
                out_right_data = right_features
            else:
                out_right_data = np.concatenate(
                    (out_right_data, right_features), axis=0)

        # print(out_left_data.shape,out_right_data.shape)

        out_left_data = np.mean(out_left_data, axis=0)
        out_right_data = np.mean(out_right_data, axis=0)
        out_left_data = out_left_data[np.newaxis, :]
        out_right_data = out_right_data[np.newaxis, :]
        D = Euclidean_distance(out_left_data, out_right_data)
        D_max = D.max()
        line_feaure = line_layer.GetFeature(fid)
        line_feaure.SetField("simi", float(D_max))
        line_layer.SetFeature(line_feaure)
        print('pair {0}: {1} {2} {3}'.format(
            i, left_polygon_id, right_polygon_id, D_max))
        # similarity_txt.write('pair {0}: {1} {2} {3}\n'.format(i,left_polygon_id,right_polygon_id ,D_max))
        break
    # similarity_txt.close()
    return 0


def MC_Lyu_2020(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    X2 = np.sum(X ** 2, axis=1)
    Y2 = np.sum(Y ** 2, axis=1)
    D = np.tile(X2.reshape(n, 1), (1, m)) + \
        (np.tile(Y2.reshape(m, 1), (1, n))).T - 2 * np.dot(X, Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D
    # return F.pairwise_distance(X, Y, p=2)


def Extract_featrues_from_multi_files(image_folder, shp_folder):
    # net =  create_model(num_classes=512,has_logits=False, is_feature_embed=True, is_multiscale_embed=True,is_label_embed = False)
    # net = ShfitScaleFormer(is_designed_feature_embedding =True, input_image_scales = [28,56,112,224] )
    # net =ShfitScaleFormer_v2(is_designed_feature_embedding =True, input_image_scales = [28,56,112,224] )
    # net = ShfitScaleFormer_v2(is_designed_feature_embedding=True, input_image_scales=[28, 56, 112])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    # net = ShfitScaleFormer_v3(is_designed_feature_embedding =False, cube_size=[8,8], input_image_scales = [32,64,128], embed_dim=768,depth=[3,2,1],)
    # net = ShfitScaleFormer_v4(is_designed_feature_embedding =False,cube_size=[8,8], input_image_scales = [32,64,128], embed_dim=768,depth=[3,2,1], )
    net =  ShfitScaleFormer_v3(is_designed_feature_embedding =True, cube_size=[8,8],input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2], )
    # net = ShfitScaleFormer_v4(is_designed_feature_embedding =True, cube_size=[8,8],input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2], )

    # net = ShfitScaleFormer_v5(depth=[6,4,2])
    # net = ShfitScaleFormer_v6()

    # print(net)
    # return
    featureIO = FeatureIO(net=net, checkpoint_path=configs.checkpoint_path)
    db_type_list = os.listdir(shp_folder)

    for dbtype in db_type_list[::]:
        if os.path.isfile(os.path.join(shp_folder, dbtype)):
            db_type_list.remove(dbtype)

    i = 0
    num_list = []
    for dbtype in db_type_list[::]:
        shp_path = r"{0}\{1}\PointsGCS.shp".format(shp_folder, dbtype)

        # city_name = dbtype[0:16]
        # tile_name = dbtype[16:]
        # name_parts = tile_name.split('_')

        # city_name = dbtype[0:16]
        # tile_name = dbtype[16:]
        name_parts = dbtype.split('_')

        # print(city_name,tile_name,name_parts)

        # break
        image_path = r"{0}\PhoenixCityGroup_{1}-{2}.tif".format(
            image_folder, name_parts[0], name_parts[1])
        # print(image_path)
        epoch_name = configs.checkpoint_path.split('_')[-1]
        epoch_name = epoch_name.split('.')[0]
        epoch_id = epoch_name[0:-6]

        h5_file_path = r"{0}\{1}\S2FormerV4-3Ch-S2E-3DP-SEF-642_{2}-epochs.h5".format(shp_folder, dbtype, epoch_id)
        # print(h5_file_path)
        # break

        # return

        # print(image_path)
        # print(shp_path)
        # print(h5_file_path)
        print("ID: {0}th / total:{1} processing {2}\\PointsGCS.shp".format(i+1, len(db_type_list) ,dbtype))
        _, patch_num = featureIO.extract_features(
            image_path, shp_path, h5_file_path)
        i += 1
        num_list.append(patch_num)
        # break
    featureIO.Close()
    for i in range(0, len(num_list)):
        print(i, num_list[i])


def Extract_featrues(image_list, shp_list):
    net = create_model(num_classes=512, has_logits=False,
                       is_feature_embed=True, is_multiscale_embed=True)
    print(net)
    featureIO = FeatureIO(net=net, checkpoint_path=configs.checkpoint_path)

    shp_folder = r"F:\01Papers\mypaper6\Data\extreme_test\OrgShapeFilePro"
    i = 0
    num_list = []
    for i in range(0, len(shp_list)):
        shp_path = shp_list[i]
        # name_parts = dbtype.split('_')
        # print(name_parts)
        image_path = image_list[i]
        # print(image_path)

        name_parts = shp_path.split('\\')
        dbtype = name_parts[-2]

        # print(dbtype)

        h5_file_path = r"{0}\{1}\features{2}.h5".format(shp_folder, dbtype, "")
        # print(h5_file_path)
        # print(shp_path)
        # print(image_path)
        print("{0}th processing {1}\\PointsGCS.shp".format(i+1, dbtype))
        # break
        _, patch_num = featureIO.extract_features(
            image_path, shp_path, h5_file_path)
        i += 1
        num_list.append(patch_num)

    featureIO.Close()
    for i in range(0, len(num_list)):
        print(i, num_list[i])


def main(argv=None):

    # net =  create_model(num_classes=512,has_logits=False, is_feature_embed=True, is_multiscale_embed=True)
    # featureIO = FeatureIO(net = net, checkpoint_path = configs.checkpoint_path)

    # image_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images\PhoenixCityGroup05_05_2.tif"
    # point_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro\PhoenixCityGroup05_05_2\PointsGCS.shp"
    # h5_file_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro\PhoenixCityGroup05_05_2\features.h5"

    # image_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\L19\PhoenixCityGroup_05-04.tif"
    # point_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\OrgShapeFilePro\05_04_05\PointsGCS.shp"
    # h5_file_path = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\OrgShapeFilePro\05_04_05\features.h5"
    print("start")
    # featureIO.extract_features(image_path, point_path, h5_file_path)
    shp_folder = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\OrgShapeFilePro"
    # shp_folder = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    image_floder = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\L19"

    # epoch_name = configs.checkpoint_path.split('_')[-1]
    # epoch_name = epoch_name.split('.')[0]
    # epoch_id = epoch_name[0:-6]

    # print(epoch_id)
    Extract_featrues_from_multi_files(image_floder, shp_folder)

    # shp_list   = [r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\02_03\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\03_03\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\04_03\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\04_04\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\05_06\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\05_07\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\05_07a\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\05_08\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\02030303\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\04030404\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\05060507\PointsGCS.shp",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\shp-PRO\05070508\PointsGCS.shp"]

    # image_list = [r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\2-3_3-3.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\2-3_3-3.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\4-3_4-4.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\4-3_4-4.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\5-6_5-7.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\5-6_5-7.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\5-7_5-8.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\5-7_5-8.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\2-3_3-3.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\4-3_4-4.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\5-6_5-7.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\EvaluationData\Composition\5-7_5-8.tif"]

    # shp_list = [r"F:\01Papers\mypaper6\Data\extreme_test\OrgShapeFilePro\Extreme05_05\PointsGCS.shp"]
    # image_list = [r"F:\01Papers\mypaper6\Data\extreme_test\image\Extreme05_05.tif"]

    # shp_list = [r"F:\01Papers\mypaper6\Data\shapefilePro\01_01_01\PointsGCS.shp",
    #             r"F:\01Papers\mypaper6\Data\shapefilePro\02_03_01\PointsGCS.shp",
    #             r"F:\01Papers\mypaper6\Data\shapefilePro\04_05_01\PointsGCS.shp",
    #             r"F:\01Papers\mypaper6\Data\shapefilePro\05_05_02\PointsGCS.shp"]
    # image_list = [r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\L19\PhoenixCityGroup_01-01.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\L19\PhoenixCityGroup_02-03.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\L19\PhoenixCityGroup_04-05.tif",
    #               r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\L19\PhoenixCityGroup_05-05.tif"]
    # Extract_featrues(image_list,shp_list)

    print("END")


if __name__ == '__main__':
    main()
