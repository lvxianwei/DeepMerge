# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:00:02 2021

@author: lxw
"""
import os
import torch
#import torch.tensor as tensor
# from Networks import SpatiallyMmemorizedNetwork
# from MyUtils import RemoteSensingDataset
from MyUtils1 import MergingSegmensPairDataset
# from MyUtils2 import MergingSegmensPairTestDataset
# from MyUtilsTestData import MergingSegmensObjectPair
from torch.utils.data import DataLoader
import numpy as np
from config import configs
import torch.optim as optim
from Losses import Loss, MultiLoss,ClassLoss
# from torch.autograd import Variable
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from vit_model import vit_base_patch_scales_224_in21k as create_model
from callbacks import LossHistory
from nets.ShfitScaleFormer import ShfitScaleFormer,ShfitScaleFormer_v2,ShfitScaleFormer_v3,ShfitScaleFormer_v4,ShfitScaleFormer_v5,ShfitScaleFormer_v6

# from osgeo import ogr

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.switch_backend('agg')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }
def show_plot(x, y, result_dir=None):
    '''
    x: iteration_number
    y: train loss
    result_dir: path to save the loss fig
    '''
    plt.plot(x, y, color='b')
    plt.legend(loc='lower right', prop=font1, frameon=False)
    plt.xlabel('Iteraion')
    plt.ylabel('Loss')
    figs_dir = os.path.join(result_dir, 'figs')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    figs_path = os.path.join(figs_dir, 'loss.jpg')
    plt.savefig(figs_path)

#如果没有该文件夹，则创建该文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        

def compute_mean_std(target_img_size, image_path, point_patch, image_statistic_path):
    
    image_statistic_txt = open(image_statistic_path, 'w')  # 写入统计信息
    img_h, img_w = target_img_size, target_img_size


    image_folder    = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images"
    polygon_folder  = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    point_folder    = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    positive_folder = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\PositiveData"
    negative_folder = r"F:\03Data\MyData\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\NegativeData"
    train_dataset = MergingSegmensPairDataset(image_folder, polygon_folder,point_folder,  positive_folder, negative_folder)    
    band_num = train_dataset.band_num
    imgs = np.zeros([img_w, img_h, band_num, 1])
    means, stdevs = [], []

    batch_size = 10
    dataloader = DataLoader( train_dataset,  batch_size = batch_size, shuffle=True)    
    for i_batch,data in enumerate(dataloader):
        print(i_batch)
        inner_img, obj_img, scene_img = data   
        inner_img = inner_img.permute(2,3,1,0)
        obj_img = obj_img.permute(2,3,1,0)
        scene_img = scene_img.permute(2,3,1,0)        
        if i_batch == 0:
            imgs = inner_img
            imgs = np.concatenate((imgs, obj_img),axis = 3)
            imgs = np.concatenate((imgs, scene_img),axis = 3)
        else:
            imgs = np.concatenate((imgs, inner_img),axis = 3)
            imgs = np.concatenate((imgs, obj_img),axis = 3)
            imgs = np.concatenate((imgs, scene_img),axis = 3)
#        im = Image.fromarray(bands)
#        im.save("cv_scaling.tif")
        # if (i_batch+1)*batch_size >= test_num:
        #     break
#    print("imgs size:",torch.tensor(imgs).size())
    
    
    imgs = imgs.astype(np.float32) / 255.#归一化
    for i in range(band_num):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

#    print("normMean = {}".format(means))
#    print("normStd = {}".format(stdevs))
    image_statistic_txt.write('normMean:{}\n'.format(means))
    image_statistic_txt.write('normStd:{}\n'.format(stdevs))
    image_statistic_txt.close()
    return means, stdevs    
 
def Euclidean_distance(X, Y):
    """
    输入：
    X n*p数组, p为特征维度
    Y m*p数组
    
    输出：
    D n*m距离矩阵
    """
    n = X.shape[0]
    m = Y.shape[0]
    X2 = np.sum(X ** 2, axis=1)
    Y2 = np.sum(Y ** 2, axis=1)
    D = np.tile(X2.reshape(n, 1), (1, m)) + (np.tile(Y2.reshape(m, 1), (1, n))).T - 2 * np.dot(X, Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D    

def Initial_one_hot(batch_size, class_num, y):
    
    one_hot = torch.zeros(batch_size,class_num,dtype=torch.long)
    
    for i in range(0, len(y)):
        # print(one_hot[i])
        one_hot[i][y[i]]=1
    # print(one_hot)
    return one_hot

def train(net, margin, train_bs, lr_init, normMean, normStd, lamda, belta, 
          is_retrained = False, checkpoint_path = None):
    
    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    
    # train_dataset = RemoteSensingDataset(configs.image_path, configs.points_path, is_train = True, train_num= train_num)
    image_folder    = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\Images"
    polygon_folder  = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    point_folder    = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\ShapeFilesPro"
    positive_folder = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\PositiveData"
    negative_folder = r"E:\01paper\A_PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\NegativeData"


    # raise("test")
    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    start_epoch = 0
    if torch.cuda.is_available():
        net.cuda()
        print("Network is uploaded into GPUs")
    if is_retrained == True: 
        model_CKPT = torch.load(checkpoint_path)
        
        # for k, v in model_CKPT['net'].items():
        #     print(k)
        # print( model_CKPT['net']['mlp.fc1.weight'])
        
        
        
        net.load_state_dict(model_CKPT['net'])
        start_epoch = model_CKPT['epoch'] + 1
        if start_epoch >= configs.num_epochs:
            raise Exception("start_epoch must be smaller than number of epochs")
#        previous_time = model_CKPT['time']
    else:
        weights = ''#'./vit_base_patch16_224_in21k.pth'
        device = torch.device('cuda:0')
        if weights != "":
            # assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
            weights_dict = torch.load(weights, map_location=device)
        #     # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if net.has_logits \
                else ['patch_embed.proj.weight', 'patch_embed.proj.bias','pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(net.load_state_dict(weights_dict, strict=False))
  

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------
    pg = [p for p in net.parameters() if p.requires_grad] 
    optimizer = optim.Adam(pg, lr=lr_init)  # 选择优化器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.2)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.2)

    if is_retrained == True: 
        optimizer.load_state_dict(model_CKPT['optimizer'])
    criterion = Loss(margin=margin, lamda=lamda, belta=belta)  
    # multi_criterion = MultiLoss(margin=margin, lamda=lamda, belta=belta) 
    # class_criterion = ClassLoss(margin=margin, lamda=lamda, belta=belta)
    # ------------------------------------ step 4/5 : 训练 ---------------------------------------
    previous_time = 0.0
    iteration_history_train = []
    loss_history_train = []
    start_time = time.time()
    print('训练开始')
    train_dataset = []
    # with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1} / {Epochs}', postfix=dict, mininterval=0.3) as pbar:
    loss_history = LossHistory('logs/')
    
    for epoch in range(start_epoch, configs.num_epochs):
        lr_temp = optimizer.param_groups[0]['lr']
        print("peoch",epoch,"-learning rate:", lr_temp)
        total_loss = 0
            
        print("Generate training data")
        train_dataset  = MergingSegmensPairDataset(image_folder, polygon_folder,point_folder,  positive_folder, negative_folder)
        
        dataset_length = len(train_dataset)
        train_loader = DataLoader(train_dataset,  batch_size = train_bs, shuffle = True)    
        train_bar = tqdm(train_loader)

        print(dataset_length, train_dataset.positive_pair_number, train_dataset.negative_pair_number)
        # for i, data in enumerate(train_loader):
        for i, data in enumerate(train_bar):

            net.train()
#            print(epoch,i)

            # 获取图片和标签
            # left_meta_data, right_meta_data, flag, left_labels, right_labels = data
            left_meta_data, right_meta_data, flag = data
            
            left_designed_features, left_scales, left_patches = left_meta_data
            right_designed_features, right_scales,right_patches = right_meta_data
            if torch.cuda.is_available():
                left_designed_features = left_designed_features.cuda()      
                left_patches0 = left_patches[0].cuda()
                left_patches1 = left_patches[1].cuda()
                left_patches2 = left_patches[2].cuda()
                left_patches3 = left_patches[3].cuda()
                left_patches = [left_patches0,left_patches1,left_patches2,left_patches3]
                # left_patches = [left_patches0,left_patches1,left_patches2]
                # left_patches = [left_patches0,left_patches1]


                right_designed_features = right_designed_features.cuda()
                right_patches0 = right_patches[0].cuda()
                right_patches1 = right_patches[1].cuda()
                right_patches2 = right_patches[2].cuda()
                right_patches3 = right_patches[3].cuda()
                right_patches = [right_patches0,right_patches1,right_patches2,right_patches3]
                # right_patches = [right_patches0,right_patches1,right_patches2]
                # right_patches = [right_patches0,right_patches1]


                flag = flag.cuda()
            # print(left_patches[2].shape)
            # break
            # print(left_designed_features.shape)
            out_a, out_b = net(left_patches,
                               left_designed_features,
                               right_patches,
                               right_designed_features)   
            
            
            # a1, left_logits, a3 = out_a
            # b1, right_logits, b3 = out_b
            # print(a1.shape, a2.shape, a3.shape)
            # print(a2)
            # left_labels = left_labels.cuda()
            # right_labels = right_labels.cuda()
            # print(a2.shape)
            # left_one_hot = Initial_one_hot(configs.batch_size,11,left_labels).cuda()
            # print(left_one_hot.shape)
            
            # print(b2.shape)
            # right_one_hot = Initial_one_hot(configs.batch_size,11,right_labels).cuda()
            # print(right_one_hot.shape)
            # print(out_a,out_b)
            
            loss = criterion(out_a, out_b, flag)
            
            # main_loss = criterion(out_a[0], out_b[0], flag)
            # aux1_loss = criterion(out_a[1], out_b[1], flag)
            # aux2_loss = criterion(out_a[2], out_b[2], flag)
            # loss = main_loss + 0.1 * aux1_loss + 0.2* aux2_loss
            
                 
            # print(loss)
            
            # multi_loss = multi_criterion(a1, b1, flag, left_logits, left_labels, right_logits, right_labels)
            # class_loss = class_criterion(left_logits, left_labels, right_logits, right_labels)
            # print(multi_loss)
            # break
            # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} time:{:.3f}".format(epoch + 1,
                                                                     configs.num_epochs,
                                                                     loss,
                                                                     round(time.time()-start_time,2)
                                                                     )
            train_bar.update(1)
            # if (i + 1) % 10 == 0:
            #     print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] lr:{} Loss: {:.4f} Time: {:.2f}".format(
            #         epoch + 1, configs.num_epochs, i + 1, len(train_loader), lr_temp, loss.data, round(time.time()-start_time,2)))
            
            #     loss_history_train.append(loss.data.cpu())
            #     iteration_history_train.append(i+1 + epoch *dataset_length )
            
        end_time = time.time()
        real_time = time.localtime()    
        if (epoch + 1) % 5 == 0 or epoch + 1 >= 90:
            model_name = "model-{0}-{1}-{2}_{3}-{4}_{5}epochs.pth".format(
                      real_time.tm_year, 
                      real_time.tm_mon, 
                      real_time.tm_mday,
                      real_time.tm_hour,
                      real_time.tm_min,
                      epoch + 1)
            state = {"net":net.state_dict(),
                     "optimizer":optimizer.state_dict(),
                     "epoch": epoch,
                     "time": round(end_time - start_time + previous_time, 2),
                     "scales": net.input_image_scales,
                     "depth": net.depth,
                     "name":net.name}
            if  epoch + 1  == 100:
                model_name = "model-{0}-{1}-{2}_{3}-{4}-{5}_{6}epochs.pth".format(
                          real_time.tm_year, 
                          real_time.tm_mon, 
                          real_time.tm_mday,
                          real_time.tm_hour,
                          real_time.tm_min,
                          net.name,
                          configs.num_epochs)
                torch.save(state, configs.model_paras_path + model_name) 
            else:
                torch.save(state, configs.model_paras_path + model_name) 
        
        if (epoch + 1) % 1 == 0:
            total_time = end_time - start_time
            remained_time = total_time * configs.num_epochs / (epoch + 1) - total_time
            print('总共的时间为:', round(total_time, 2),'secs')
            print('预计剩余时间为:', round(remained_time, 2),'secs')
            loss_history.append_loss(total_loss / len(train_loader), total_loss / len(train_loader), round(total_time, 2))
        scheduler.step()
    print('训练结束')
    end_time = time.time()
    print('总共的时间为:', round(end_time - start_time + previous_time, 2),'secs')
    print ("Finished!")
    return iteration_history_train, loss_history_train

def main(argv = None):
        
#    means, stdevs = compute_mean_std(configs.target_img_size, configs.image_path, configs.points_path, configs.image_statistic_path)
#    print(means, stdevs)
    
    # net =  create_model(num_classes=512,has_logits=False, is_feature_embed=True, is_multiscale_embed=True,is_label_embed = False)
    # net = ShfitScaleFormer(is_designed_feature_embedding =False, input_image_scales = [28,56,112,224])
    # net = ShfitScaleFormer_v2(is_designed_feature_embedding =True, input_image_scales = [28,56,112,224])
    # net = ShfitScaleFormer_v2(is_designed_feature_embedding =True, input_image_scales = [28,56,112])
    # net = ShfitScaleFormer_v2(is_designed_feature_embedding =True, input_image_scales = [32,64,128],cube_size=[8,8])
    # net = ShfitScaleFormer_v3(is_designed_feature_embedding =True,input_image_scales = [32,64,128] )
    # net = ShfitScaleFormer_v4(is_designed_feature_embedding =False,depth = [3,1,1])
    # net1 = ShfitScaleFormer_v4(is_designed_feature_embedding =True, cube_size=[8,8], input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2], )
    # net2 = ShfitScaleFormer_v4(is_designed_feature_embedding =False, cube_size=[8,8],input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2], )
    
    #注意是否注释掉了 embedding
    net = ShfitScaleFormer_v3(is_designed_feature_embedding =True, cube_size=[8,8], input_image_scales = [32,64,128], embed_dim=768,depth=[6,4,2],)
    # net = ShfitScaleFormer_v2(is_designed_feature_embedding =False, cube_size=[8,8],input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2], )
    # net = ShfitScaleFormer_v4(is_designed_feature_embedding =True, cube_size=[8,8],input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2], )

    # net = ShfitScaleFormer_v5(depth=[6,4,2])
    # net = ShfitScaleFormer_v6()
    margin = 1.0
    train_bs = configs.batch_size
    lr_init = 1e-4
    normMean = 0.0
    normStd = 0.0
    lamda = 0.1
    belta = 0
    
    x, y =train(net, margin, train_bs, lr_init, normMean, normStd, lamda, belta, 
                is_retrained = True, checkpoint_path = configs.checkpoint_path )
    
    print("main function is running")
  
if __name__ == '__main__':
    main()
    
    
    