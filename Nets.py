# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:41:05 2021

@author: lxw
"""
import torch
from torch import optim, nn
import torch.nn.functional as F
import math
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        
        # self.model = nn.Sequential(
        #     nn.Linear(784,250),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(250,250),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(250,10),
        #     nn.LeakyReLU(inplace=True),
        #     )
        
        self.fc1 = nn.Linear(784,250)
        self.fc2 = nn.Linear(250,250)
        self.fc3 = nn.Linear(250,10)
        
    def forward(self, x):
        fc1_map = self.fc1(x)
        fc1_map = F.leaky_relu(fc1_map)
        fc2_map = self.fc2(fc1_map)
        fc2_map = F.leaky_relu(fc2_map)
        fc3_map = self.fc3(fc2_map)
        fc3_map = F.leaky_relu_(fc3_map)
        return fc3_map,fc2_map
    
class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.fc3 = nn.Linear(250,10)
        
    def forward(self, x):
        fc3_map = self.fc3(x)
        fc3_map = F.leaky_relu_(fc3_map)
        return fc3_map
    

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
 
        # self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
        #     input_size=28,
        #     hidden_size=80,         # rnn hidden unit
        #     num_layers=4,           # number of rnn layer
        #     batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        #     bidirectional= True
        # )
        
        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=28,
            hidden_size=80,         # rnn hidden unit
            num_layers=4,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional= True
        )
        self.dropout = nn.Dropout(0.5)
        
 
        self.out = nn.Linear(160, 10)
        
    def attention_net(self, x, query, mask=None): 
        
        d_k = query.size(-1)     # d_k为query的维度
        # print("d_k = ",d_k)
       
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        # print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        # print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1) 
        # print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)
        # print("context: ", context.size())
        return context, alpha_n 
 
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        r_out, h_n = self.rnn(x, None)
        # print("r_out = ",r_out.size())
        # choose r_out at the last time step
        # out = self.out(r_out[:, -1, :])
        # print(out.size())
        # r_out = r_out.permute(1, 0, 2)
        print(r_out.size())
        query = self.dropout(r_out)
        print("query = ", query.size())
        attn_output, alpha_n = self.attention_net(r_out, query)
        # print("attn_output = ", attn_output.size())
        out = self.out(attn_output)
        # print("out = ", out.size())
        return out

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            