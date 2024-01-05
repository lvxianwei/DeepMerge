# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:16:28 2021

@author: lxw
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
import torchvision.models as models
import numpy as np
from Non_local_block import NONLocalBlock2D
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

class SpatiallyMmemorizedNetwork(nn.Module):
    def __init__(self, base_net,pooling, in_channels, out_channels, reduced_size):
        super(SpatiallyMmemorizedNetwork, self).__init__()
#        if base_net == "AlexNet":
#            alexnet = models.alexnet(pretrained=True)
#            
#        elif base_net == "VGG16":
#            vgg16 = models.vgg16(pretrained=True)
#            
#        elif base_net == "VGG19":
#            vgg19 = models.vgg19(pretrained=True)
#        
#        elif base_net == "ResNet50":
#            resnet50 = models.resnet50(pretrained=True)
#            
#        elif base_net == "InceptionV3":
#            inceptionV3 = models.inception_v3(pretrained=True)
#        
#        else:
#            raise ValueError('Please use a specified CNN!')
        
        vgg16 = models.vgg16(pretrained=True)
      
#        vgg16 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(
                vgg16.features,
                NONLocalBlock2D(in_channels = 512),
                nn.AdaptiveAvgPool2d((1,1))
                )
        
#        self.lstm=nn.LSTM(
#            input_size = 512,  #每行的像素点
#            hidden_size = 256,
#            num_layers = 2,  # the num of RNN layers
#            batch_first=True, # 表示在输入数据的时候，数据的维度为(batch,time_step,input)
#                               #如果自己定义的维度为(time_step,batch,input),则为False
#            bidirectional = True
#        )
        
        # self.dropout = nn.Dropout(0.5)
        
        self.reduce_conv = None
        if reduced_size < out_channels:
            print('Feature size reduction: {} -> {}'.format(out_channels, reduced_size))
            self.reduce_conv = nn.Conv2d(out_channels, reduced_size, (1, 1))
        
        self.pooling = pooling
        self.eps = 1e-6
        
#     def attention_net(self, x, query, mask=None): 
        
#         d_k = query.size(-1)     # d_k为query的维度
# #        print("d_k = ",d_k)
       
#         # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
#         # print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
#         # 打分机制 scores: [batch, seq_len, seq_len]
#         scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
# #        print("x", x.shape)
# #        print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
#         # 对最后一个维度 归一化得分
#         alpha_n = F.softmax(scores, dim=-1) 
# #        print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
#         # 对权重化的x求和
#         # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
#         context = torch.matmul(alpha_n, x).sum(1)
# #        print("context: ", context.size())
#         return context, alpha_n 
        
    def forward_once(self, x):
#        print("input x size:",x.shape)
        x = self.features(x)
#        print("feature x size:", x.shape)
#        x = F.adaptive_avg_pool2d(x, (1,1)) #没用？？？
#        print("avg_pool x size:",x.shape)
        if self.reduce_conv is not None:
            x = self.reduce_conv(x)
#        print("x:",x.shape)
        x = x.squeeze(3)#去掉维度为1的维度
        x = x.squeeze(2)
        
#        print("squeeze x:",x.shape)
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x) #需要注意输入的图像的batch size = 1的情况
        
        return x
    
    def forward_twice(self, x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        return y1, y2

    def forward_thrice1(self, x1, x2, x3):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        y3 = self.forward_once(x3)
        return y1, y2, y3





    def forward_thrice(self, x1, x2, x3):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        y3 = self.forward_once(x3)
      
        y1 = y1[:,:, np.newaxis]
        y2 = y2[:,:, np.newaxis]
        y3 = y3[:,:, np.newaxis]
        y1 = y1.permute(0,2,1) 
        y2 = y2.permute(0,2,1)
        y3 = y3.permute(0,2,1)
#        print("y3 shape",y3.shape)
        y = torch.cat((y1,y2,y3),dim = 1)
#        print("cat y shape",y.shape)
#        r_out, h_n = self.lstm(y)
#        r_out = r_out.contiguous()
#        query = self.dropout(r_out)
#        attn_output, alpha_n = self.attention_net(r_out, query)
#        return attn_output
        
        query = self.dropout(y)
#        print("drop y", query.shape)
        y, alpha_n = self.attention_net(y, query)
        y = y.contiguous()
#        print("y shape",y.shape)
        y = y.view(-1, 512)
#        y = y.view(-1, 1536)
        return y
    
    def forward_sixice(self,x11, x12, x13, x21, x22, x23):
        y1 = self.forward_thrice(x11, x12, x13)
        y2 = self.forward_thrice(x21, x22, x23)
        return y1, y2
        
    
    def forward_nince(self, x11, x12, x13, x21, x22, x23, x31, x32, x33):
        y1 = self.forward_thrice(x11, x12, x13)
        y2 = self.forward_thrice(x21, x22, x23)
        y3 = self.forward_thrice(x31, x32, x33)
        return y1, y2, y3
        
    
    def forward(self, *args):
        n = len(args)
        if n == 1:
            return self.forward_once(args[0])
        elif n == 2:
            return self.forward_twice(args[0], args[1])
        elif n == 3:
            return self.forward_thrice1(args[0], args[1], args[2])#这里的forward有两种！！！！注意
        elif n == 6:
            return self.forward_nince(args[0], args[1], args[2],args[3], args[4], args[5])
        elif n == 9:
            return self.forward_nince(args[0], args[1], args[2],args[3], args[4], args[5],args[6], args[7], args[8])
        else:
            raise ValueError('Invalid input arguments! You got {} arguments.'.format(n))
            
            
def test():        
    batch_size = 1            
    x1 = torch.ones(batch_size,3,224,224).cuda()
    x2 = torch.ones(batch_size,3,224,224).cuda()
    x3 = torch.ones(batch_size,3,224,224).cuda()
    
    x4 = torch.ones(batch_size,3,224,224).cuda()
    x5 = torch.ones(batch_size,3,224,224).cuda()
    x6 = torch.ones(batch_size,3,224,224).cuda()
    
    x7 = torch.ones(batch_size,3,224,224).cuda()
    x8 = torch.ones(batch_size,3,224,224).cuda()
    x9 = torch.ones(batch_size,3,224,224).cuda()
    net = SpatiallyMmemorizedNetwork("VGG16","GAP",3,512,512)
    print(net)
    net.cuda()
    #net.cuda()
    # y1, y2, y3 = net(x1,x2,x3,x4, x5, x6, x7, x8, x9)
    y1, y2, y3 = net(x1,x2,x3)
    #y = net(x1)
    y1 = y1.cpu()
    y2 = y2.cpu()
    y3 = y3.cpu()
    y12 = torch.cat((y1,y2),dim=0) 
#    print("y12 shape:", y12.shape)
#    #length = len(y)
    print("y1 shape:",y1.shape)
    print("y2 shape:",y2.shape)
    print("y3 shape:",y3.shape)
#    print(y12)
#    print(y1)
#    print(y2)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum( p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
#    print("y3 shape:",y3.shape)


if __name__ == "__main__":
    test()
















