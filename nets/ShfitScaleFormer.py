# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:18:29 2023

@author: lxw
"""
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_,DropPath
import math

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, out_c=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, out_c, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(out_c) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
      
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   
    
class FeatureEmbed(nn.Module):
    def __init__(self, feature_size = 19, embed_dim=768,act_layer=nn.GELU, norm_layer=None):
        super().__init__()

        self.proj0 =  nn.Conv1d(in_channels = feature_size, out_channels = embed_dim, kernel_size = 1, stride = 1)
        self.proj1 =  nn.Conv1d(in_channels = embed_dim, out_channels = embed_dim, kernel_size = 1, stride = 1)
        self.proj2 =  nn.Conv1d(in_channels = embed_dim, out_channels = embed_dim, kernel_size = 1, stride = 1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.act = act_layer()
    def forward(self, x):
        #B 
        # print("x shape:", x.shape)
        B, C, W = x.shape
        x = x.permute(0, 2, 1)
        # assert H == self.img_size[0] and W == self.img_size[1], \
            # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj0(x)
        x = self.act(x)
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x

class CrossScaleAttention(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 cube_size,# 要编码的块的3D大小,其中 x*y*z == input.shape[1]
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0):
        super(CrossScaleAttention, self).__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.cube_size = cube_size  # Wh, Ww        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * cube_size[0] - 1) * (2 * cube_size[1] - 1) * (2 * cube_size[2] - 1), num_heads)
            )  #(2*Wh-1)*(2*Ww-1)*(2*Wc-1), nH
        self.initial_relative_position_index(cube_size= self.cube_size) #生成3D相对位置索引
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*Batch, Number of patches, Channels of Dim (Embedding dim))
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.cube_size[0] * self.cube_size[1]* self.cube_size[2], 
            self.cube_size[0] * self.cube_size[1]* self.cube_size[2], 
            -1)  # Wh*Ww,Wh*Ww,nH
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) #3D相对位置编码
        # attn = attn 
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  
        

    def initial_relative_position_index(self,cube_size):
        # get pair-wise relative position index for each token inside the window
        coords_c = torch.arange(self.cube_size[0]) #层 lay  [0, 1, 2,..., lay -1] c
        coords_h = torch.arange(self.cube_size[1]) #行 row  [0, 1, 2,..., row -1] h
        coords_w = torch.arange(self.cube_size[2]) #列 col  [0, 1, 2,..., col -1] w
        
        coords = torch.stack(torch.meshgrid([coords_c, coords_h, coords_w]))# 3, Wh, Ww, Wc组成坐标系, 返回横坐标集合,纵坐标集合,竖坐标集合
        coords_flatten = torch.flatten(coords,1)  # 2, Wh*Ww 

        relative_coords = coords_flatten[:, :,None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2 contiguous改变内存顺序
        relative_coords[:, :, 0] += self.cube_size[0] - 1  # z axis  shift to start from 0
        relative_coords[:, :, 1] += self.cube_size[1] - 1  # y axis
        relative_coords[:, :, 2] += self.cube_size[2] - 1  # x axis
        relative_coords[:, :, 1] *= 2 * self.cube_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.cube_size[1] - 1)*(2 * self.cube_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
class CrossScaleBlock(nn.Module):        

    def __init__(self, 
                 dim, 
                 num_heads, 
                 cube_size, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_ratio=0., 
                 attn_drop_ratio=0., 
                 drop_path_ratio=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossScaleAttention(dim = dim, num_heads=num_heads, cube_size = cube_size,qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)   
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class CrossScaleAttention_v5(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads,
                 cube_size,# 要编码的块的3D大小,其中 x*y*z == input.shape[1]
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0):
        super(CrossScaleAttention_v5, self).__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.cube_size = cube_size  # Wh, Ww        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * cube_size[0] - 1) * (2 * cube_size[1] - 1) * (2 * cube_size[2] - 1) +cube_size[2] * cube_size[1] * cube_size[0] * 2 , 
                num_heads)
            )  #(2*Wh-1)*(2*Ww-1)*(2*Wc-1), nH
        # print("cube",self.cube_size)
        self.initial_relative_position_index(cube_size= self.cube_size) #生成3D相对位置索引
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def initial_relative_position_index(self,cube_size):
        # get pair-wise relative position index for each token inside the window
        coords_c = torch.arange(self.cube_size[0]) #层 lay  [0, 1, 2,..., lay -1] c
        coords_h = torch.arange(self.cube_size[1]) #行 row  [0, 1, 2,..., row -1] h
        coords_w = torch.arange(self.cube_size[2]) #列 col  [0, 1, 2,..., col -1] w
        # print(self.cube_size)

        coords = torch.stack(torch.meshgrid([coords_c, coords_h, coords_w]))# 3, Wh, Ww, Wc组成坐标系, 返回横坐标集合,纵坐标集合,竖坐标集合
        coords_flatten = torch.flatten(coords,1)  # 2, Wh*Ww 

        relative_coords = coords_flatten[:, :,None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2 contiguous改变内存顺序
        relative_coords[:, :, 0] += self.cube_size[0] - 1  # z axis  shift to start from 0
        relative_coords[:, :, 1] += self.cube_size[1] - 1  # y axis
        relative_coords[:, :, 2] += self.cube_size[2] - 1  # x axis
        relative_coords[:, :, 1] *= 2 * self.cube_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.cube_size[1] - 1)*(2 * self.cube_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # print(relative_coords.shape)
        # print("pos index:",relative_position_index.shape)
        # print(relative_position_index)
        max_id = torch.max(relative_position_index)
        # print("max id:", max_id)
        # print(len(relative_position_index[0]))
        col = torch.full([len(relative_position_index[0]),1],max_id+1)
        for i in range(0,len(col)):
            col[i] = max_id + i + 1
        # print("row shape",col.shape)

        relative_position_index = torch.cat([relative_position_index,col],dim=1)
        # print("relative_position_index shape",relative_position_index.shape)
        max_id = torch.max(relative_position_index)

        row =  torch.full([1,len(relative_position_index[1])],max_id+1)
        # print("row shape:",row.shape)
        for i in range(0, len(row[0])):
            row[0][i] = max_id + i + 1
            
        # print(row)
            

        relative_position_index = torch.cat([relative_position_index,row],dim=0)
        # print("relative_position_index shape",relative_position_index.shape)
        # print(relative_position_index)

        relative_position_index[-1][-1] = relative_position_index[0][0]
        # print(relative_position_index)
        # print(len(self.relative_position_bias_table))
        
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*Batch, Number of patches, Channels of Dim (Embedding dim))
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # print("id shape:", self.relative_position_index.shape)
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.cube_size[0] * self.cube_size[1]* self.cube_size[2] + 1, 
            self.cube_size[0] * self.cube_size[1]* self.cube_size[2] + 1, 
            -1)  # Wh*Ww,Wh*Ww,nH
        
        # print("bias shape:",relative_position_bias.shape)
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  
        
class CrossScaleBlock_v5(nn.Module):        

    def __init__(self, 
                 dim, 
                 num_heads, 
                 cube_size, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_ratio=0., 
                 attn_drop_ratio=0., 
                 drop_path_ratio=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossScaleAttention_v5(dim = dim, num_heads=num_heads, cube_size = cube_size,qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)   
        
    def forward(self, x):
        # print("Cross block x shsape:", x.shape)
        # atten = self.attn(self.norm1(x))
        # print("Cross block attn shsape:", atten.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
  
class AuxBolck(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, in_c=768, out_c=100, cube_size= [3,8,8], norm_layer=nn.LayerNorm):
        super().__init__()
        self.cube_size = cube_size
        self.aux = nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=2, padding=0, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.3),
        nn.Conv2d(in_c, int(in_c / cube_size[0]) , kernel_size=1)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = norm_layer(in_c)
        self.out_features = nn.Linear(in_c, out_c) 


    def forward(self, x):
        
        # B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        y = []
        for i in range(0, self.cube_size[0]):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.aux(x_data)
            x_data = x_data.flatten(2)
            x_data = self.avgpool(x_data)
            x_data = torch.flatten(x_data, 1)#去除最后的一维无用向量
            y.append(x_data)
            
            # break
        x = torch.cat(tuple(y),1)
        x = self.norm(x)
        x = self.out_features(x)
        return x         
    
class AuxBolck_v5(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, in_c=768, out_c=100, cube_size= [3,8,8], norm_layer=nn.LayerNorm):
        super().__init__()
        self.cube_size = cube_size
        self.aux = nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=2, padding=0, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.3),
        nn.Conv2d(in_c, int(in_c / cube_size[0]) , kernel_size=1)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = norm_layer(in_c *2)
        self.out_features = nn.Linear(in_c * 2, out_c) 


    def forward(self, x):
        
        # B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        y = []
        for i in range(0, self.cube_size[0]):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.aux(x_data)
            x_data = x_data.flatten(2)
            x_data = self.avgpool(x_data)
            x_data = torch.flatten(x_data, 1)#去除最后的一维无用向量
            y.append(x_data)
    
            # break
        x_last_data = x[:, int(self.cube_size[1] * self.cube_size[2] * (i + 1)):,:].transpose(1, 2)
        x_last_data = torch.flatten(x_last_data,1)
        # print("x_last_data shape:",x_last_data.shape)
        # y.append(x_last_data)
        x = torch.cat(tuple(y),1)
        x = torch.cat([x,x_last_data],1)
        # print("aux shape:",x.shape)
        # x = self.norm(x)
        x = self.out_features(x)
        return x  
#支持相对尺度编码                
class ShfitScaleFormer(nn.Module):
    def __init__(self,
                 num_classes=11,
                 is_designed_feature_embedding = True,
                 FeatureEmbed = FeatureEmbed, 
                 PatchEmbed = PatchEmbed, 
                 cube_size = [7,7],
                 input_image_scales = [28,56,112,224],
                 embed_dim=768,
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0,
                 drop_path_ratio=0.,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 cuda = True):
        super(ShfitScaleFormer, self).__init__()
        self.num_classes = num_classes
        self.is_designed_feature_embedding = is_designed_feature_embedding
        self.patch_embed_layer = PatchEmbed
        self.feature_embed_layer = FeatureEmbed
        self.input_image_scales = input_image_scales
        self.input_scales_num = len(input_image_scales)
        self.cube_size = cube_size
        self.cube_size.insert(0, self.input_scales_num)        
        self.num_features = int(self.input_scales_num * embed_dim)
    
    
        # self.patch_embed_blocks = nn.ModuleList()
        # for i in range(self.input_scales_num):
        #     patch_embed = self.patch_embed_layer(img_size=self.input_image_scales[i],  patch_size=int(self.input_image_scales[i] /  self.cube_size[1]),  in_c=3, out_c=768)
        #     self.patch_embed_blocks.append(patch_embed)
        
        

        self.patch_embed_scale0 = self.patch_embed_layer(img_size=self.input_image_scales[0],  patch_size=4,  in_c=3, out_c=768)
        self.patch_embed_scale1 = self.patch_embed_layer(img_size=self.input_image_scales[1],  patch_size=8,  in_c=3, out_c=768)
        self.patch_embed_scale2 = self.patch_embed_layer(img_size=self.input_image_scales[2],  patch_size=16, in_c=3, out_c=768)
        self.patch_embed_scale3 = self.patch_embed_layer(img_size=self.input_image_scales[3],  patch_size=32, in_c=3, out_c=768)
        
        
        
        self.feature_embed = self.feature_embed_layer(feature_size = 19, embed_dim=768) if is_designed_feature_embedding else None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= self.cube_size,
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=dpr[i], 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.norm = norm_layer([196,768])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.final_features = nn.Linear(int(self.input_scales_num * embed_dim), 100) 
        # self.mlp = Mlp(in_features=self.num_features, hidden_features=self.num_features,out_features = self.num_features,drop = 0.3) 
        self.final_features_with_design = nn.Linear(int((self.input_scales_num + 1) * embed_dim), 100) 
 
        self.head = nn.Linear(100, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.apply(self._init_weights)
    
    def patch_embed(self,x):
        x0 = self.patch_embed_scale0(x[0])
        x1 = self.patch_embed_scale1(x[1])
        x2 = self.patch_embed_scale2(x[2])
        x3 = self.patch_embed_scale3(x[3])
        x = torch.cat((x0, x1, x2, x3),1)

        # y = []
        # for i in range(0, self.input_scales_num):
        #     y.append(self.patch_embed_blocks[i](x[i]))
        # x = torch.cat(tuple(y),1)

        return x
    
    def designed_feature_embed(self,x):
        x = self.feature_embed(x)
        return x
        
    def forward_once_design_feature(self,x, designed_features):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x =self.norm(x)
        # y = []
        # for i in range(0, self.input_scales_num):
        #     x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
        #     x_data = self.avgpool(x_data.transpose(1, 2)) 
        #     x_data = torch.flatten(x_data, 1)
        #     y.append(x_data)
        x0 = x[:,0:49,:]
        x1 = x[:,49:98,:]
        x2 = x[:,98:147,:]
        x3 = x[:,147:196,:]
        x0 = self.avgpool(x0.transpose(1, 2)) 
        x1 = self.avgpool(x1.transpose(1, 2)) 
        x2 = self.avgpool(x2.transpose(1, 2)) 
        x3 = self.avgpool(x3.transpose(1, 2)) 
        x0 = torch.flatten(x0, 1)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)
        designed_features = self.designed_feature_embed(designed_features)
        designed_features = torch.squeeze(designed_features, dim=1)
        designed_features = self.norm(designed_features)
        x = torch.cat((x0,x1,x2,x3,designed_features),1)        
        # x = torch.cat(tuple(y),1)
        # x = torch.cat(x,designed_features,1)
        x = self.final_features_with_design(x)
        
        return x
        
    def forward_once(self,x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x =self.norm(x)
        
        # y = []
        # for i in range(0, self.input_scales_num):
        #     x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
        #     x_data = self.avgpool(x_data.transpose(1, 2)) 
        #     x_data = torch.flatten(x_data, 1)
        #     y.append(x_data)
        # x = torch.cat(tuple(y),1)
        x0 = x[:,0:49,:]
        x1 = x[:,49:98,:]
        x2 = x[:,98:147,:]
        x3 = x[:,147:196,:]
        x0 = self.avgpool(x0.transpose(1, 2)) 
        x1 = self.avgpool(x1.transpose(1, 2)) 
        x2 = self.avgpool(x2.transpose(1, 2)) 
        x3 = self.avgpool(x3.transpose(1, 2)) 
        x0 = torch.flatten(x0, 1)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)
        x = torch.cat((x0,x1,x2,x3),1)
       
        x = self.final_features(x)
        return x
     
    def extract_features_with_design_features(self,x_path, x_designed_features):
         return self.forward_once_design_feature(x_path, x_designed_features)
     
    def extract_features(self,x_path):
         return self.forward_once(x_path)
                          
        
    def forward(self,
                x1_patches, x1_designed_features,
                x2_patches = None, x2_designed_features = None):
        if  x1_designed_features != None and x2_patches == None and x2_designed_features == None:
            return self.extract_features_with_design_features(x1_patches, x1_designed_features)
        
        if x1_designed_features == None and x2_patches == None and x2_designed_features == None:
            return self.extract_features(x1_patches, x1_designed_features)
        
        if self.is_designed_feature_embedding == False:
            # 'x1 x2'
            feature1 = self.forward_once(x1_patches)
            feature2 = self.forward_once(x2_patches)
            return  feature1, feature2
        else:
            feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
            feature2 = self.forward_once_design_feature(x2_patches, x2_designed_features)
            
            return feature1,feature2



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)        

#支持相对尺度编码，尺度可选可变                     
class ShfitScaleFormer_v2(nn.Module):
    def __init__(self,
                 num_classes=11,
                 is_designed_feature_embedding = True,
                 FeatureEmbed = FeatureEmbed, 
                 PatchEmbed = PatchEmbed, 
                 cube_size = [7,7],
                 input_image_scales = [28,56,112,224],
                 embed_dim=768,
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0,
                 drop_path_ratio=0.,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 cuda = True):
        super(ShfitScaleFormer_v2, self).__init__()
        self.num_classes = num_classes
        self.is_designed_feature_embedding = is_designed_feature_embedding
        self.patch_embed_layer = PatchEmbed
        self.feature_embed_layer = FeatureEmbed
        self.input_image_scales = input_image_scales
        self.input_scales_num = len(input_image_scales)
        self.cube_size = cube_size
        self.cube_size.insert(0, self.input_scales_num)        
        self.num_features = int(self.input_scales_num * embed_dim)
    
        self.patch_embed_blocks = nn.ModuleList()
        for i in range(self.input_scales_num):
            patch_embed = self.patch_embed_layer(img_size=self.input_image_scales[i],  patch_size=int(self.input_image_scales[i] /  self.cube_size[1]),  in_c=3, out_c=768)
            self.patch_embed_blocks.append(patch_embed)
        
        self.feature_embed = self.feature_embed_layer(feature_size = 19, embed_dim=768) if is_designed_feature_embedding else None
        
        # dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= self.cube_size,
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(12)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.norm = norm_layer([196,768])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.final_features = nn.Linear(int(self.input_scales_num * embed_dim), 100) 
        # self.mlp = Mlp(in_features=self.num_features, hidden_features=self.num_features,out_features = self.num_features,drop = 0.3) 
        self.final_features_with_design = nn.Linear(int((self.input_scales_num + 1) * embed_dim), 100) 
 
        self.head = nn.Linear(100, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.apply(self._init_weights)
    
    def patch_embed(self,x):

        y = []
        i = 0
        for layer in self.patch_embed_blocks:
            y.append(layer(x[i]))
            i += 1
        x = torch.cat(tuple(y),1)




        return x
    
    def designed_feature_embed(self,x):
        x = self.feature_embed(x)
        return x
        
    def forward_once_design_feature(self,x, designed_features):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x =self.norm(x)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = self.avgpool(x_data.transpose(1, 2)) 
            x_data = torch.flatten(x_data, 1)
            y.append(x_data)
        designed_features = self.designed_feature_embed(designed_features)
        designed_features = torch.squeeze(designed_features, dim=1)
        designed_features = self.norm(designed_features)
        # x = torch.cat((x0,x1,x2,x3,designed_features),1)        
        x = torch.cat(tuple(y),1)
        x = torch.cat((x,designed_features),1)
        x = self.final_features_with_design(x)
        
        return x
        
    def forward_once(self,x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x =self.norm(x)
        
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = self.avgpool(x_data.transpose(1, 2)) 
            x_data = torch.flatten(x_data, 1)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.final_features(x)
        return x
     
    def extract_features_with_design_features(self,x_path, x_designed_features):
         return self.forward_once_design_feature(x_path, x_designed_features)
     
    def extract_features(self,x_path):
         return self.forward_once(x_path)
                          
        
    def forward(self,
                x1_patches, x1_designed_features,
                x2_patches = None, x2_designed_features = None):
        if self.training:
            if self.is_designed_feature_embedding == False:
            # 'x1 x2'
                feature1 = self.forward_once(x1_patches)
                feature2 = self.forward_once(x2_patches)
                return  feature1, feature2
            else:
                
                feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
                feature2 = self.forward_once_design_feature(x2_patches, x2_designed_features)
            
                return feature1,feature2
        if self.eval:
            if self.is_designed_feature_embedding == False:
            # 'x1 x2'
                feature1 = self.forward_once(x1_patches)
                return  feature1
            else:
                feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
                return feature1




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)         

#支持相对尺度编码，尺度可选可变，参数减少         
class ShfitScaleFormer_v3(nn.Module):
    def __init__(self,
                 num_classes=11,
                 is_designed_feature_embedding = True,
                 FeatureEmbed = FeatureEmbed, 
                 PatchEmbed = PatchEmbed, 
                 cube_size = [8,8],
                 input_image_scales = [32,64,128],
                 embed_dim=768,
                 depth=[6,4,2], 
                 num_heads=12, 
                 mlp_ratio=4.0,
                 drop_path_ratio=0.,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 cuda = True):
        super(ShfitScaleFormer_v3, self).__init__()
        self.name = "S2Former_v3-3CH"
        
        if is_designed_feature_embedding == True:
            self.name = "{0}-3DP-SEF".format( self.name)
        self.name = "{0}-{1}{2}{3}".format(self.name,depth[0],depth[1],depth[2])
        print(self.name)
        self.num_classes = num_classes
        self.is_designed_feature_embedding = is_designed_feature_embedding
        self.patch_embed_layer = PatchEmbed
        self.feature_embed_layer = FeatureEmbed
        self.input_image_scales = input_image_scales
        self.input_scales_num = len(input_image_scales)
        self.cube_size = cube_size
        self.cube_size.insert(0, self.input_scales_num)        
        self.num_features = int(self.input_scales_num * embed_dim)
        self.depth = depth
        self.patch_embed_blocks = nn.ModuleList()
        for i in range(self.input_scales_num):
            patch_embed = self.patch_embed_layer(img_size=self.input_image_scales[i],  patch_size=int(self.input_image_scales[i] /  self.cube_size[1]),  in_c=3, out_c=768)
            self.patch_embed_blocks.append(patch_embed)
        
        self.feature_embed = self.feature_embed_layer(feature_size = 19, embed_dim=768) if is_designed_feature_embedding else None
        
        # dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks0 = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= self.cube_size,
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth[0])
        ])
        
        
        self.blocks1 = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= [self.input_scales_num, 4, 4],
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth[1])
        ])
        
        
        self.blocks2 = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= [self.input_scales_num, 2, 2],
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth[2])
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.norm = norm_layer([196,768])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.final_features = nn.Linear(int(self.input_scales_num * embed_dim), 100) 
        # self.mlp = Mlp(in_features=self.num_features, hidden_features=self.num_features,out_features = self.num_features,drop = 0.3) 
        self.final_features_with_design = nn.Linear(int((self.input_scales_num + 1) * embed_dim), 100) 
 
        self.head = nn.Linear(100, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool2D = nn.AvgPool2d(kernel_size=2, stride=2)
        self.apply(self._init_weights)
    
    def patch_embed(self,x):

        y = []
        i = 0
        for layer in self.patch_embed_blocks:
            # print(i)
            y.append(layer(x[i]))
            i += 1
        x = torch.cat(tuple(y),1)




        return x
    
    def designed_feature_embed(self,x):
        x = self.feature_embed(x)
        return x
        
    def backbone(self,x):
        
        x = self.blocks0(x)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.avgpool2D(x_data) 
            x_data = x_data.flatten(2)
            x_data = x_data.transpose(1, 2)
            # print("X pool 3", x_data.shape)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.norm(x)
        x = self.blocks1(x)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, 4 * 4 * i : 4 *4 * (i + 1),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.avgpool2D(x_data) 
            x_data = x_data.flatten(2)
            x_data = x_data.transpose(1, 2)
            # print("X pool 3", x_data.shape)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.norm(x)
        x = self.blocks2(x)
        
        
        return x
    
    
    def forward_once_design_feature(self,x, designed_features):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.backbone(x)
        x =self.norm(x)

        # print(x.shape)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, 2 * 2* i : 2* 2 * (i + 1),:]
            # print(x_data.shape)
            x_data = self.avgpool(x_data.transpose(1, 2)) 
            x_data = torch.flatten(x_data, 1)
            # print(x_data.shape)

            y.append(x_data)
        x = torch.cat(tuple(y),1)
        designed_features = self.designed_feature_embed(designed_features)
        designed_features = torch.squeeze(designed_features, dim=1)
        designed_features = self.norm(designed_features)
        # x = torch.cat((x0,x1,x2,x3,designed_features),1)        
        # print(x.shape)

        x = torch.cat((x,designed_features),1)
        # print(x.shape)

        x = self.final_features_with_design(x)
        
        return x
    

        
    def forward_once(self,x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.backbone(x)
        x =self.norm(x)
        
        y = []
        for i in range(0, self.input_scales_num):
            x_data =  x[:, 2 * 2* i : 2* 2 * (i + 1),:]
            x_data = self.avgpool(x_data.transpose(1, 2)) 
            x_data = torch.flatten(x_data, 1)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.final_features(x)
        return x
     
    def extract_features_with_design_features(self,x_path, x_designed_features):
         return self.forward_once_design_feature(x_path, x_designed_features)
     
    def extract_features(self,x_path):
         return self.forward_once(x_path)
                          
        
    def forward(self,
                x1_patches, x1_designed_features,
                x2_patches = None, x2_designed_features = None):
        if self.training:
            if self.is_designed_feature_embedding == False:
            # 'x1 x2'
                feature1 = self.forward_once(x1_patches)
                feature2 = self.forward_once(x2_patches)
                return  feature1, feature2
            else:
                
                feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
                feature2 = self.forward_once_design_feature(x2_patches, x2_designed_features)
            
                return feature1,feature2
        if self.eval:
            if self.is_designed_feature_embedding == False:
            # 'x1 x2'
                feature1 = self.forward_once(x1_patches)
                return  feature1
            else:
                feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
                return feature1



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)     
        
#支持相对尺度编码，尺度可选可变，参数减少，辅助loss
class ShfitScaleFormer_v4(nn.Module):
    def __init__(self,
                 num_classes=11,
                 is_designed_feature_embedding = True,
                 FeatureEmbed = FeatureEmbed, 
                 PatchEmbed = PatchEmbed, 
                 cube_size = [8,8],
                 input_image_scales = [32,64,128],
                 embed_dim=768,
                 depth=[3,2,1], 
                 num_heads=12, 
                 mlp_ratio=4.0,
                 drop_path_ratio=0.,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 cuda = True):
        super(ShfitScaleFormer_v4, self).__init__()
        self.name = "S2Former_v4-3CH"
        
        if is_designed_feature_embedding == True:
            self.name = "{0}-SFE".format( self.name)
        self.name = "{0}-{1}{2}{3}".format(self.name,depth[0],depth[1],depth[2])
        print(self.name)
        self.num_classes = num_classes
        self.is_designed_feature_embedding = is_designed_feature_embedding
        self.patch_embed_layer = PatchEmbed
        self.feature_embed_layer = FeatureEmbed
        self.input_image_scales = input_image_scales
        self.input_scales_num = len(input_image_scales)
        self.cube_size = cube_size
        self.cube_size.insert(0, self.input_scales_num)        
        self.num_features = int(self.input_scales_num * embed_dim)
        self.depth = depth
        self.patch_embed_blocks = nn.ModuleList()
        for i in range(self.input_scales_num):
            patch_embed = self.patch_embed_layer(img_size=self.input_image_scales[i],  patch_size=int(self.input_image_scales[i] /  self.cube_size[1]),  in_c=3, out_c=768)
            self.patch_embed_blocks.append(patch_embed)
        
        self.feature_embed = self.feature_embed_layer(feature_size = 19, embed_dim=768) if is_designed_feature_embedding else None
        
        # dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks0 = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= self.cube_size,
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(self.depth[0])
        ])
        
        
        self.blocks1 = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= [self.input_scales_num, 4, 4],
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(self.depth[1])
        ])
        
        
        self.blocks2 = nn.Sequential(*[
            CrossScaleBlock(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= [self.input_scales_num, 2, 2],
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(self.depth[2])
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.norm = norm_layer([196,768])
        self.final_features = nn.Linear(int(self.input_scales_num * embed_dim), 100) 
        self.final_features_with_design = nn.Linear(int((self.input_scales_num + 1) * embed_dim), 100) 
        self.head = nn.Linear(100, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool2D = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.aux0 = AuxBolck()
        self.aux1 = AuxBolck(cube_size= [3,4,4])

        
        
        self.apply(self._init_weights)
    
    def patch_embed(self,x):

        y = []
        i = 0
        for layer in self.patch_embed_blocks:
            y.append(layer(x[i]))
            i += 1
        x = torch.cat(tuple(y),1)
        return x
    
    def designed_feature_embed(self,x):
        x = self.feature_embed(x)
        x = torch.squeeze(x, dim=1)
        x = self.norm(x)
        return x
        
    def backbone(self,x):
        
        x = self.blocks0(x)
        aux0 = self.aux0(x)
        # print("aux0 shape:", aux0.shape)
        
        
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.avgpool2D(x_data) 
            x_data = x_data.flatten(2)
            x_data = x_data.transpose(1, 2)
            # print("X pool 3", x_data.shape)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.norm(x)
        
        x = self.blocks1(x)
        aux1 = self.aux1(x)
        # print("aux1 shape:", aux1.shape)

        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, 4 * 4 * i : 4 *4 * (i + 1),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.avgpool2D(x_data) 
            x_data = x_data.flatten(2)
            x_data = x_data.transpose(1, 2)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.norm(x)
        x = self.blocks2(x)
        x =self.norm(x)
        # print(x.shape)

        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, 2 * 2* i : 2* 2 * (i + 1),:]
            # print(i,"shape:",x_data.shape)
            x_data = x_data.transpose(1, 2)
            x_data = self.avgpool(x_data) 
            # print(i,"shape:",x_data.shape)
            x_data = torch.flatten(x_data, 1)#去除最后的一维无用向量
            # print(i,"shape:",x_data.shape)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        # print(x.shape)

        return x, aux0, aux1
    
    
    def forward_once_design_feature(self,x, designed_features):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x, aux0, aux1 = self.backbone(x)
        
        designed_features = self.designed_feature_embed(designed_features)
        x = torch.cat((x,designed_features),1)
        # print(x.shape)

        x = self.final_features_with_design(x)
        
        if self.training:
            return x, aux0, aux1
        
        if self.eval:
            return x
        return None
    

        
    def forward_once(self,x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x, aux0, aux1 = self.backbone(x)
        x = self.final_features(x)
        
        if self.training:
            return x, aux0, aux1
        
        if self.eval:
            return x
     
    def extract_features_with_design_features(self,x_path, x_designed_features):
         return self.forward_once_design_feature(x_path, x_designed_features)
     
    def extract_features(self,x_path):
         return self.forward_once(x_path)
                          
        
    def forward(self,
                x1_patches, x1_designed_features,
                x2_patches = None, x2_designed_features = None):
        # if  x1_designed_features != None and x2_patches == None and x2_designed_features == None:
        #     return self.extract_features_with_design_features(x1_patches, x1_designed_features)
        
        # if x1_designed_features == None and x2_patches == None and x2_designed_features == None:
        #     return self.extract_features(x1_patches, x1_designed_features)
        if self.training:
            if self.is_designed_feature_embedding == False:
            # 'x1 x2'
                feature1 = self.forward_once(x1_patches)
                feature2 = self.forward_once(x2_patches)
                return  feature1, feature2
            else:
                
                feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
                feature2 = self.forward_once_design_feature(x2_patches, x2_designed_features)
            
                return feature1,feature2
        if self.eval:
            if self.is_designed_feature_embedding == False:
            # 'x1 x2'
                feature1 = self.forward_once(x1_patches)
                return  feature1
            else:
                feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
                return feature1



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)     
  
#支持相对尺度编码，尺度可选可变，参数减少，辅助loss，改变人工特征输入位置
class ShfitScaleFormer_v5(nn.Module):
    def __init__(self,
                 num_classes=11,
                 FeatureEmbed = FeatureEmbed, 
                 PatchEmbed = PatchEmbed, 
                 CrossScaleBlock = CrossScaleBlock_v5,
                 AuxBolck = AuxBolck_v5,
                 cube_size = [8,8],
                 input_image_scales = [32,64,128],
                 embed_dim=768,
                 depth=[3,2,1], 
                 num_heads=12, 
                 mlp_ratio=4.0,
                 drop_path_ratio=0.,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 cuda = True):
        super(ShfitScaleFormer_v5, self).__init__()
        self.num_classes = num_classes
        self.patch_embed_layer = PatchEmbed
        self.feature_embed_layer = FeatureEmbed
        self.cross_scale_block = CrossScaleBlock
        self.aux_block = AuxBolck_v5
        self.input_image_scales = input_image_scales
        self.input_scales_num = len(input_image_scales)
        self.cube_size = cube_size
        self.cube_size.insert(0, self.input_scales_num)        
        self.num_features = int(self.input_scales_num * embed_dim)
        self.depth = depth
        self.patch_embed_blocks = nn.ModuleList()
        for i in range(self.input_scales_num):
            patch_embed = self.patch_embed_layer(img_size=self.input_image_scales[i],  patch_size=int(self.input_image_scales[i] /  self.cube_size[1]),  in_c=3, out_c=768)
            self.patch_embed_blocks.append(patch_embed)
        
        self.feature_embed = self.feature_embed_layer(feature_size = 19, embed_dim=768)
        
        # dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks0 = nn.Sequential(*[
            self.cross_scale_block(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= self.cube_size,
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth[0])
        ])
        
        
        self.blocks1 = nn.Sequential(*[
            self.cross_scale_block(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= [self.input_scales_num, 4, 4],
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth[1])
        ])
        
        
        self.blocks2 = nn.Sequential(*[
            self.cross_scale_block(dim=embed_dim, 
                            num_heads=num_heads, 
                            cube_size= [self.input_scales_num, 2, 2],
                            mlp_ratio=mlp_ratio,
                            drop_ratio=drop_ratio, 
                            attn_drop_ratio=attn_drop_ratio,
                            drop_path_ratio=0, 
                            norm_layer=norm_layer, 
                            act_layer=act_layer)
            for i in range(depth[2])
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.norm = norm_layer([196,768])
        self.final_features = nn.Linear(int(self.input_scales_num * embed_dim), 100) 
        self.final_features_with_design = nn.Linear((2 * embed_dim), 100) 
        self.last_block_features = nn.Linear(int((self.input_scales_num + 1) * embed_dim), embed_dim) 

        self.head = nn.Linear(100, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool2D = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.aux0 = self.aux_block()
        self.aux1 = self.aux_block(cube_size= [3,4,4])

        
        
        self.apply(self._init_weights)
    
    def patch_embed(self,x):

        y = []
        i = 0
        for layer in self.patch_embed_blocks:
            y.append(layer(x[i]))
            i += 1
        x = torch.cat(tuple(y),1)
        return x
    
    def designed_feature_embed(self,x):
        x = self.feature_embed(x)
        x = torch.squeeze(x, dim=1)
        x = self.norm(x)
        return x
        
    def backbone(self,x):
        
        # print("backbone x input shape:",x.shape)
        x = self.blocks0(x)
        aux0 = self.aux0(x)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.avgpool2D(x_data) 
            x_data = x_data.flatten(2)
            x_data = x_data.transpose(1, 2)
            # print("X pool 3", x_data.shape)
            y.append(x_data)
        
        y.append(x[:, int(self.cube_size[1] * self.cube_size[2] * (i + 1)):,:])
        x = torch.cat(tuple(y),1)
        # print("")
        # print("backbone x0 shape:".x.shape)
        x = self.norm(x)
        x = self.blocks1(x)
        aux1 = self.aux1(x)
        # print("aux1 shape:", aux1.shape)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, 4 * 4 * i : 4 *4 * (i + 1),:]
            x_data = x_data.transpose(1, 2)
            x_data = x_data.reshape([x_data.shape[0],x_data.shape[1], int(math.sqrt(x_data.shape[2])), int(math.sqrt(x_data.shape[2])) ])           
            x_data = self.avgpool2D(x_data) 
            x_data = x_data.flatten(2)
            x_data = x_data.transpose(1, 2)
            y.append(x_data)
        y.append(x[:, 4 *4 *  self.input_scales_num:,:])
        x = torch.cat(tuple(y),1)
        x = self.norm(x)
        x = self.blocks2(x)
        x =self.norm(x)
        # print(x.shape)
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, 2 * 2* i : 2* 2 * (i + 1),:]
            # print(i,"shape:",x_data.shape)
            x_data = x_data.transpose(1, 2)
            x_data = self.avgpool(x_data) 
            # print(i,"shape:",x_data.shape)
            x_data = torch.flatten(x_data, 1)#去除最后的一维无用向量
            # print(i,"shape:",x_data.shape)
            y.append(x_data)
            
        x_data = x[:, 2* 2 * self.input_scales_num:,:].transpose(1, 2)
        x_data = torch.flatten(self.avgpool(x_data),1)
        # print(x_data.shape)
        y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.last_block_features(x)
        # print(x.shape)
        return x, aux0, aux1
    
    
    def forward_once_design_feature(self,x, designed_features):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        # print("Shape:",x.shape)
        designed_features = self.designed_feature_embed(designed_features)
        SFE_features = torch.unsqueeze(designed_features, dim = 1)
        fixed_x = torch.cat((x,SFE_features),1)
        # print("fixed shape:",fixed_x.shape)        
        x, aux0, aux1 = self.backbone(fixed_x)
        # print(x.shape,designed_features.shape)
        x = torch.cat((x,designed_features),1)
        x = self.final_features_with_design(x)
        if self.training:
            return x, aux0, aux1
        
        if self.eval:
            return x
        
    def forward_once(self,x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.backbone(x)
        x =self.norm(x)
        
        y = []
        for i in range(0, self.input_scales_num):
            x_data = x[:, int(self.cube_size[1] * self.cube_size[2] * i) : int(self.cube_size[1] * self.cube_size[2] * (i + 1)),:]
            x_data = self.avgpool(x_data.transpose(1, 2)) 
            x_data = torch.flatten(x_data, 1)
            y.append(x_data)
        x = torch.cat(tuple(y),1)
        x = self.final_features(x)
        return x
     
    def extract_features_with_design_features(self,x_path, x_designed_features):
         return self.forward_once_design_feature(x_path, x_designed_features)
     
    def extract_features(self,x_path):
         return self.forward_once(x_path)
                          
        
    def forward(self,
                x1_patches, x1_designed_features,
                x2_patches = None, x2_designed_features = None):
        if  x1_designed_features != None and x2_patches == None and x2_designed_features == None:
            return self.extract_features_with_design_features(x1_patches, x1_designed_features)
        
        if x1_designed_features == None and x2_patches == None and x2_designed_features == None:
            return self.extract_features(x1_patches, x1_designed_features)
        

        feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
        feature2 = self.forward_once_design_feature(x2_patches, x2_designed_features)
            
        return feature1,feature2



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)      
        
#人工特征网络
class ShfitScaleFormer_v6(nn.Module):
    def __init__(self,
                 num_classes=11,
                 FeatureEmbed = FeatureEmbed, 
                 embed_dim=768,
                 mlp_ratio=4.0,
                 drop_path_ratio=0.,
                 drop_ratio=0., 
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 cuda = True):
        super(ShfitScaleFormer_v6, self).__init__()
        self.num_classes = num_classes
        self.feature_embed_layer = FeatureEmbed

        self.feature_embed = self.feature_embed_layer(feature_size = 19, embed_dim=768)  

        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.norm = norm_layer([196,768])
        self.final_features_with_design = nn.Linear(embed_dim, 100) 
        self.head = nn.Linear(100, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def designed_feature_embed(self,x):
        x = self.feature_embed(x)
        x = torch.squeeze(x, dim=1)
        x = self.norm(x)
        return x
    
    
    def forward_once_design_feature(self,x, designed_features):
        
        designed_features = self.designed_feature_embed(designed_features)
        designed_features = self.final_features_with_design(designed_features)
        return designed_features

    def extract_features_with_design_features(self,x_path, x_designed_features):
         return self.forward_once_design_feature(x_path, x_designed_features)
        
    def forward(self,
                x1_patches, x1_designed_features,
                x2_patches = None, x2_designed_features = None):
        if  x1_designed_features != None and x2_patches == None and x2_designed_features == None:
            return self.extract_features_with_design_features(x1_patches, x1_designed_features)
        
        if x1_designed_features == None and x2_patches == None and x2_designed_features == None:
            return self.extract_features(x1_patches, x1_designed_features)
        

        feature1 = self.forward_once_design_feature(x1_patches, x1_designed_features)
        feature2 = self.forward_once_design_feature(x2_patches, x2_designed_features)    
        return feature1,feature2



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)     

    
def initial_relative_position_index(cube_size):
    # get pair-wise relative position index for each token inside the window
    coords_c = torch.arange(cube_size[0]) #层 lay  [0, 1, 2,..., lay -1] c
    coords_h = torch.arange(cube_size[1]) #行 row  [0, 1, 2,..., row -1] h
    coords_w = torch.arange(cube_size[2]) #列 col  [0, 1, 2,..., col -1] w
    # coords = torch.stack(torch.meshgrid([coords_c, coords_h]))# 3, Wh, Ww, Wc组成坐标系, 返回横坐标集合,纵坐标集合,竖坐标集合

    coords = torch.stack(torch.meshgrid([coords_c, coords_h, coords_w]))# 3, Wh, Ww, Wc组成坐标系, 返回横坐标集合,纵坐标集合,竖坐标集合
    print(coords)
    coords_flatten = torch.flatten(coords,1)  # 2, Wh*Ww 
    print(coords_flatten)
    relative_coords = coords_flatten[:, :,None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2 contiguous改变内存顺序
    relative_coords[:, :, 0] += cube_size[0] - 1  # z axis  shift to start from 0
    relative_coords[:, :, 1] += cube_size[1] - 1  # y axis
    
    # relative_coords[:, :, 0] *= (2 * cube_size[1] - 1)

    relative_coords[:, :, 2] += cube_size[2] - 1  # x axis
    relative_coords[:, :, 1] *= 2 * cube_size[2] - 1
    relative_coords[:, :, 0] *= (2 * cube_size[1] - 1)*(2 * cube_size[2] - 1)
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    
    # print(relative_position_index)
    
    # print(relative_coords)

def main():
    print('main function start')
    # [batch_size, num_patches, total_embed_dim]
    # x = torch.randn(4,196,768)# [B, 196, 768]
    # cube_size = [4,7,7]   
    # win_att = CrossScaleAttention( dim = 768, cube_size = cube_size, num_heads = 12 )
    # out = win_att(x)
    
    # x = torch.randn(16,196,768)# [B, 196, 768]
    # cube_size = [4,7,7]   
    # win_att = CrossScaleBlock( dim = 768, num_heads = 12, cube_size = cube_size, )
    # out = win_att(x)
    # print(out.shape)
    
    
    
    
    # left_patches0 = torch.randn(4,3,28,28).cuda()
    # left_patches1 = torch.randn(4,3,56,56).cuda()
    # left_patches2 = torch.randn(4,3,112,112).cuda()
    # left_patches3 = torch.randn(4,3,224,224).cuda()
    
    
    # left_patches0 = torch.randn(4,3,32,32).cuda()
    # left_patches1 = torch.randn(4,3,64,64).cuda()
    # left_patches2 = torch.randn(4,3,128,128).cuda()
    # left_patches3 = torch.randn(4,3,256,256).cuda()
    
    
    # features = torch.randn(4,1,19).cuda()
    
    
    # x1 = [left_patches0,left_patches1,left_patches2,left_patches3]
    # x2 = [left_patches0,left_patches1,left_patches2,left_patches3]
    
    
    # x1 = [left_patches0,left_patches1,left_patches2]
    # x2 = [left_patches0,left_patches1,left_patches2]
    # S2Former = ShfitScaleFormer(is_designed_feature_embedding = True, input_image_scales = [28,56,112,224])
    # S2Former = ShfitScaleFormer_v2(is_designed_feature_embedding = True, input_image_scales = [28,56,112,224])
    # S2Former = ShfitScaleFormer_v2(is_designed_feature_embedding = True, input_image_scales = [28,56,112])
    # S2Former = ShfitScaleFormer_v2(is_designed_feature_embedding =False, input_image_scales = [32,64,128],cube_size=[8,8])
    # S2Former = ShfitScaleFormer_v3(is_designed_feature_embedding =True, cube_size=[8,8])
    # S2Former = ShfitScaleFormer_v4(is_designed_feature_embedding =True, cube_size=[8,8])
    # S2Former = ShfitScaleFormer_v5()
    # S2Former = ShfitScaleFormer_v6()
    # S2Former = ShfitScaleFormer_v4(is_designed_feature_embedding =False, cube_size=[8,8],input_image_scales = [32,64,128,256], embed_dim=768, depth=[6,4,2], )
    # S2Former = ShfitScaleFormer_v4(is_designed_feature_embedding =False, cube_size=[8,8],input_image_scales = [32,64,128], embed_dim=768, depth=[6,4,2] )



    # print(S2Former)
    # S2Former.cuda()
    # S2Former.train()
    # S2Former.eval()

    # out = S2Former(x1,features,x2,features)
    # print(out[0].shape, out[1].shape)
    
    cube_size = [2,2,2]
    
    initial_relative_position_index(cube_size)
    print('mian function end')
if __name__ == "__main__":
    main()
    
    
    

    
    
    
    
    
    
    
    