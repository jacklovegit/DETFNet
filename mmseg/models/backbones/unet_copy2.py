import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
#123456
#365489
#针对下采样阶段
from mycode.MDFA import MDFA  #原创模块涨点了
from mycode.HWD import Down_wt

#---针对上采样阶段
from mycode.DySample import DySample  #涨点了，替换了双线性插值
from mycode.BFM import BFM  #原创模块涨点了
from mycode.MECS import MECS
#------
#???
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvModule(in_channels, mid_channels, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(mid_channels, out_channels, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

    def forward(self, x):
        return self.double_conv(x)
#原来的Down
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()

        if bilinear:
            #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
            # self.conv = nn.Conv2d(in_channels//2, out_channels, 3, 1,1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bfm = BFM(in_channels // 2)
        self.mecs = MECS(in_channels // 2, in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = self.bfm(x1, x2)
        x = self.mecs(x)
        return self.conv(x)

#DySample+BFM,这里特征融合主要是通道数设置为in_channels // 2
# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, bilinear=True, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
#         super().__init__()
#
#         if bilinear:
#             #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.up = DySample(in_channels)
#             self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
#             # self.conv = nn.Conv2d(in_channels//2, out_channels, 3, 1,1)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels // 2, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
#         #self.pagfm = PagFM(out_channels, out_channels)
#         self.bfm = BFM(in_channels // 2)
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
#                                     diffY // 2, diffY - diffY // 2])
#         #x = self.pagfm(x2, x1)
#         x = self.bfm(x1, x2)
#         return self.conv(x)

#MECS
# class Up(nn.Module):
#     """Upscaling then double conv with MECS"""
#     def __init__(self, in_channels, out_channels, bilinear=True, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
#         super().__init__()
#
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#         # 添加 MECS 模块
#         self.attention = MECS(in_channels, in_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)  # 上采样
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)  # 在通道维度上连接上采样后的 x1 和 x2
#
#         # 应用 MECS 注意力机制
#         x = self.attention(x)
#
#         return self.conv(x)


#Dysample+MECS
# class Up(nn.Module):
#     """Upscaling then double conv with MECS"""
#     def __init__(self, in_channels, out_channels, bilinear=True, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
#         super().__init__()
#
#         if bilinear:
#             #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.up = DySample(in_channels)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#         # 添加 MECS 模块
#         self.attention = MECS(in_channels, in_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)  # 上采样
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)  # 在通道维度上连接上采样后的 x1 和 x2
#         # 应用 MECS 注意力机制
#         x = self.attention(x)
#
#         return self.conv(x)

#  DySample+BFM+MECS
# class Up(nn.Module):
#     """Upscaling then double conv with MECS"""
#     def __init__(self, in_channels, out_channels, bilinear=True, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
#         super().__init__()
#
#         if bilinear:
#             #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.up = DySample(in_channels)
#             self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
#             # self.conv = nn.Conv2d(in_channels//2, out_channels, 3, 1,1)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels // 2, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.bfm = BFM(in_channels // 2)
#         self.mecs = MECS(in_channels // 2, in_channels // 2)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
#                                     diffY // 2, diffY - diffY // 2])
#         #x = self.pagfm(x2, x1)
#         x = self.bfm(x1, x2)
#
#         # 应用 MECS 注意力机制
#         x = self.mecs(x)
#
#         return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

@MODELS.register_module()
class myunet(BaseModule):
    def __init__(self, n_channels,
                 n_classes,
                 bilinear=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 ):
        super(myunet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, norm_cfg=norm_cfg, act_cfg=act_cfg)
        #在这这里加
        self.down1 = Down(64, 128, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.down2 = Down(128, 256, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.down3 = Down(256, 512, norm_cfg=norm_cfg, act_cfg=act_cfg)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up1 = Up(1024, 512 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up2 = Up(512, 256 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up3 = Up(256, 128 // factor, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up4 = Up(128, 64, bilinear, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):#torch.Size([4, 3, 64, 64])
        # print(x.shape)
        x1 = self.inc(x) #torch.Size([4, 64, 64, 64])
        #----还有在这加
        # print(x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        dec_outs = [x5]
        # print(x5.size())
        # print(x4.size())
        x = self.up1(x5, x4)
        # print(x.size())
        dec_outs.append(x)
        x = self.up2(x, x3)
        dec_outs.append(x)
        x = self.up3(x, x2)
        dec_outs.append(x)
        x = self.up4(x, x1)
        dec_outs.append(x)

        # logits = self.outc(x)
        return dec_outs



