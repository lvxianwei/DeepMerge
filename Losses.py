# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 09:51:03 2021

@author: lxw
"""


from torch import nn
import torch.nn.functional as F

class Loss(nn.Module):
    
    def __init__(self, margin, lamda, belta):
        super(Loss,self).__init__()
        self.margin = margin
        self.lamda = lamda
        self.belta = belta
        
#     def forward(self, anchor, positive, negative, size_average=True):
#         distance_positive = (anchor - positive).pow(2).sum(1)
#         distance_negative = (anchor - negative).pow(2).sum(1)
#         distance_negative_1 = (positive - negative).pow(2).sum()
        
#         losses1 = F.relu(distance_positive - distance_negative + self.margin) #relu这里的作用相当于与0比较求max
#         losses2 = F.relu(distance_positive - distance_negative_1 + self.margin)
#         losses3 = self.lamda * F.relu(distance_positive - self.belta)
        
#         #losses = losses1 + losses2
#         losses = losses1 + losses2 + losses3
# #        losses = losses1
        # return losses.mean() if size_average else losses.sum()
    
    def forward(self, positive, negative, flag, size_average=True):

        distance = (positive - negative).pow(2).sum(1)
        losses = flag*distance + (1- flag)*F.relu(-distance + self.margin)
        return losses.mean()
    
    
class MultiLoss(nn.Module):
     
    def __init__(self, margin, lamda, belta):
        super(MultiLoss,self).__init__()
        self.margin = margin
        self.lamda = lamda
        self.belta = belta
        
    def Loss_Class(self, inputs, targets):
        
        # print(inputs.shape,targets.shape)
        # print(inputs,targets)
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(inputs,targets)
        return loss
         
     
    def forward(self, positive, negative, flag, left_logits, left_one_hot, right_logits, right_one_hot, size_average=True):

        distance = (positive - negative).pow(2).sum(1)
        losses = flag*distance + (1- flag)*F.relu(-distance + self.margin)
        
        
        loss1 = self.Loss_Class(left_logits, left_one_hot)
        loss2 = self.Loss_Class(right_logits, right_one_hot)
        
        total_loss = 0.7*losses.mean() + 0.15 * loss1.mean() + 0.15 * loss2.mean()
        
        return total_loss   
        
        
class ClassLoss(nn.Module):
     
    def __init__(self, margin, lamda, belta):
        super(ClassLoss,self).__init__()
        self.margin = margin
        self.lamda = lamda
        self.belta = belta
        
    def Loss_Class(self, inputs, targets):
        
        # print(inputs.shape,targets.shape)
        # print(inputs,targets)
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(inputs,targets)
        return loss
         
     
    def forward(self, left_logits, left_one_hot, right_logits, right_one_hot, size_average=True):

        loss1 = self.Loss_Class(left_logits, left_one_hot)
        loss2 = self.Loss_Class(right_logits, right_one_hot)
        
        total_loss =0.5 * loss1.mean() + 0.5 * loss2.mean()
        
        return total_loss   