import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Pred_Layer(nn.Module):
    def __init__(self, dim_in=512, dim_out=512):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),     # Anet/其他数据集512 可能需要更改
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Conv1d(dim_out, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1

class ASPP(nn.Module):
    def __init__(self, dim):
        super(ASPP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, padding=0),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x = torch.cat((x1, x2, x3), dim=1)

        return x

class DSAM(nn.Module):
    def  __init__(self, dim):
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(dim)
        self.bf_conv = ASPP(dim)
        self.rgbd_pred_layer = Pred_Layer(dim * 6, dim)

    def forward(self, feat_0, attn_map):  # P0，p4
        [_, _, T] = feat_0.size()
        attn_map = torch.sigmoid(
            F.interpolate(attn_map,
                          size=(T),
                          mode='linear',  # nearest
                          align_corners=True))  # FPAM

        ff_feat = self.ff_conv(feat_0 * attn_map)  # 下路空洞卷积模块
        bf_feat = self.bf_conv(feat_0 * (1 - attn_map))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat


class TIR_Block(nn.Module):
    """Temporal Imbalance Refinement Module"""
    def __init__(self, dim=512):
        super().__init__()
        self.rgb_global = Pred_Layer(dim, dim)
        self.dsam = DSAM(dim)

    def forward(self, x):
        x1 = list(x)
        e4, p4 = self.rgb_global(x1[5])
        fused_feat = self.dsam(x1[0], p4)
        x1[0] = fused_feat + x1[0]

        return x1
