"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import os
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
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


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward_once(self,x):
        x = self.forward_features(x)
        # print(x.shape)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    
    def forward_twice(self,x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        return y1, y2
    
    def forward_thrice(self, x1, x2, x3):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        y3 = self.forward_once(x3)
        return y1, y2, y3


    def forward(self,*args):
        n = len(args)
        if n == 1:
            return self.forward_once(args[0])
        elif n == 2:
            return self.forward_twice(args[0], args[1])
        elif n == 3:
            return self.forward_thrice(args[0], args[1], args[2])
        else:
            raise ValueError('Invalid input arguments! You got {} arguments.'.format(n))
            
            

class ScaleEmbedTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, scales = [1,1,1,1], qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., 
                 is_multiscale_embed = True,embed_layer=PatchEmbed, 
                 is_feature_embed = True,feature_embed = FeatureEmbed,
                 is_label_embed = False,
                 norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(ScaleEmbedTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.scales = scales
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.is_multiscale_embed = is_multiscale_embed
        self.patch_embed  = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim) if is_multiscale_embed == False else None
        self.patch_embed0 = embed_layer(img_size=28, patch_size=4, in_c=in_c, embed_dim=embed_dim) if is_multiscale_embed else None
        self.patch_embed1 = embed_layer(img_size=56, patch_size=8, in_c=in_c, embed_dim=embed_dim) if is_multiscale_embed else None
        self.patch_embed2 = embed_layer(img_size=112, patch_size=16, in_c=in_c, embed_dim=embed_dim) if is_multiscale_embed else None
        self.patch_embed3 = embed_layer(img_size=224, patch_size=32, in_c=in_c, embed_dim=embed_dim) if is_multiscale_embed else None
        
        
        
        # num_patches = 196# self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.is_label_embed =is_label_embed
        if self.is_label_embed == True:
            self.label_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.is_feature_embed = is_feature_embed
        self.feature_embed = feature_embed(feature_size = 19, embed_dim=768) if is_feature_embed else None
        
        
        
        self.pos_embed0 = nn.Parameter(torch.zeros(1, 49, embed_dim))
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 49, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 49, embed_dim))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 49, embed_dim))
        self.pos_embed_non_multiscale = nn.Parameter(torch.zeros(1, 196, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
            
        if is_label_embed == True:
            self.class_logits =  nn.Linear(100, 11)

        else:
            self.class_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
            
        #my head
        self.my_head = nn.Linear(768, 100)
        if is_label_embed == True:
            
            self.my_class_head =  nn.Sequential(
            nn.Linear(embed_dim, 100),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(100, 100),
            )
        else:
            # self.my_head = nn.Identity()
            self.my_class_head  = nn.Identity()
        # Weight init
        nn.init.trunc_normal_(self.pos_embed_non_multiscale, std=0.02)
        nn.init.trunc_normal_(self.pos_embed0, std=0.02)
        nn.init.trunc_normal_(self.pos_embed1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed2, std=0.02)
        nn.init.trunc_normal_(self.pos_embed3, std=0.02)
        # nn.init.trunc_normal_(self.my_head , std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.is_label_embed == True:
            nn.init.trunc_normal_(self.label_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, designed_feature):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        if self.is_multiscale_embed == True:
            x0 = self.patch_embed0(x[0])
            x0 = (x0 + self.pos_embed0) * self.scales[0]
            x1 = self.patch_embed1(x[1])
            x1 = (x1 + self.pos_embed1) * self.scales[1]
            x2 = self.patch_embed2(x[2])
            x2 = (x2 + self.pos_embed2) * self.scales[2]
            x3 = self.patch_embed3(x[3])
            x3 = (x3 + self.pos_embed3) * self.scales[3]
            
            x = torch.cat((x0, x1, x2, x3),1)
        else:
            
            x = self.patch_embed(x)
            # print(x.shape)
            # print(self.pos_embed_non_multiscale.shape)
            x = x + self.pos_embed_non_multiscale
                    
        # [1, 1, 768] -> [B, 1, 768]
        cls_token   = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.is_label_embed == True:
            label_token = self.label_token.expand(x.shape[0], -1, -1)
        
        
        if self.dist_token is None:
            if self.is_feature_embed == True:
                # print("designed feature was enbeded")
                designed_feature = self.feature_embed(designed_feature)
                x = torch.cat((cls_token,designed_feature, x), dim=1) #[B, 197, 768] [B, 198, 768]
                
                if self.is_label_embed == True:
                    x = torch.cat((cls_token,label_token ,designed_feature, x), dim=1) #[B, 197, 768] [B, 198, 768]
                
                
                
            else:
                # print("no designed feature was enbeded")
                x = torch.cat((cls_token,x), dim=1)  # [B, 197, 768] [B, 198, 768]
        else:
            x = torch.cat((cls_token,designed_feature, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            

        # print(x.shape)        
        # x = self.pos_drop(x + self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            x_label = x[:,0]
            x_label = self.my_head(x_label)
            
            if self.is_label_embed == True:
                x_class = x[:,1]
                x_class = self.my_class_head(x_class)
                return self.pre_logits(x_label), self.class_logits(x_class),x_class
            else:
                return self.pre_logits(x_label)
        else:
            return x[:, 0], x[:, 1]
            # return y, x[:, 1]

    def forward_once(self,x):
        x = self.forward_features(x, None)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # print("call head")
            # print(x.shape)
            x = self.head(x)
            # print(x.shape)
            
        return x
    
    def forward_twice(self,x1, x2):
        y = self.forward_features(x1, x2)
        return y
    
    def forward_forice(self, x1, x2, x3, x4):
        y1 = self.forward_twice(x1, x2)
        y2 = self.forward_twice(x3, x4)
        return y1, y2


    def forward(self,*args):
        n = len(args)
        if n == 1:
            return self.forward_once(args[0])
        elif n == 2:
            return self.forward_twice(args[0], args[1])
        elif n == 4:
            return self.forward_forice(args[0], args[1], args[2], args[3])
        else:
            raise ValueError('Invalid input arguments! You got {} arguments.'.format(n))

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def vit_base_patch_scales_224_in21k(num_classes: int = 21843, has_logits: bool = True, is_feature_embed = True, is_multiscale_embed = True,is_label_embed = False):
    model = ScaleEmbedTransformer(img_size=224,
                                  patch_size=16,
                                  embed_dim=768,
                                  depth=12,
                                  num_heads=12,
                                  representation_size=768 if has_logits else None,
                                  num_classes=num_classes,
                                  is_feature_embed = is_feature_embed,
                                  is_multiscale_embed = is_multiscale_embed,
                                  is_label_embed = is_label_embed)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model

def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model

def test():        
    batch_size = 10            
    x0 = torch.ones(batch_size,3,28,28).cuda()
    x1 = torch.ones(batch_size,3,56,56).cuda()
    x2 = torch.ones(batch_size,3,112,112).cuda()
    x3 = torch.ones(batch_size,3,224,224).cuda()
    feature = torch.ones(batch_size,1,19).cuda()
    x = [x0, x1, x2, x3]
    # print(x.shape)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    print(x[3].shape)
    net = vit_base_patch_scales_224_in21k(num_classes=512,has_logits=False, is_feature_embed = False, is_multiscale_embed = False)
    net.cuda()
    weights = './vit_base_patch16_224_in21k.pth'
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
    
    
    
    
    # y= net(x, feature)
    y= net(x3,feature)
    print(y.shape)
    
   
    # print(feature.shape)
    # feature_embed = FeatureEmbed(feature_size = 15, embed_dim=768)
    # feature_embed.cuda()
    # out_feature = feature_embed(feature)
    # print(out_feature.shape)


if __name__ == "__main__":
    test()



























