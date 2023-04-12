import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.warplayer import warp
from model.transformer_layers import TFModel

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )


def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FlowRefineNetA(nn.Module):
    def __init__(self, context_dim, c=16, r=1, n_iters=4):
        super(FlowRefineNetA, self).__init__()
        corr_dim = c
        flow_dim = c
        motion_dim = c
        hidden_dim = c

        self.n_iters = n_iters
        self.r = r
        self.n_pts = (r * 2 + 1) ** 2

        self.occl_convs = nn.Sequential(nn.Conv2d(2 * context_dim, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, 1, 1, 1, 0),
                                        nn.Sigmoid())

        self.corr_convs = nn.Sequential(nn.Conv2d(self.n_pts, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, corr_dim, 1, 1, 0),
                                        nn.PReLU(corr_dim))

        self.flow_convs = nn.Sequential(nn.Conv2d(2, hidden_dim, 3, 1, 1),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1),
                                        nn.PReLU(flow_dim))

        self.motion_convs = nn.Sequential(nn.Conv2d(corr_dim + flow_dim, motion_dim, 3, 1, 1),
                                          nn.PReLU(motion_dim))

        self.gru = nn.Sequential(nn.Conv2d(motion_dim + context_dim * 2 + 2, hidden_dim, 3, 1, 1),
                                 nn.PReLU(hidden_dim),
                                 nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1),
                                 nn.PReLU(flow_dim), )

        self.flow_head = nn.Sequential(nn.Conv2d(flow_dim, hidden_dim, 3, 1, 1),
                                       nn.PReLU(hidden_dim),
                                       nn.Conv2d(hidden_dim, 2, 3, 1, 1))

    def L2normalize(self, x, dim=1):
        eps = 1e-12
        norm = x ** 2
        norm = norm.sum(dim=dim, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x/norm)

    def forward_once(self, x0, x1, flow0, flow1):
        B, C, H, W = x0.size()

        x0_unfold = F.unfold(x0, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H,
                                                                               W)  # (B, C*n_pts, H, W)
        x1_unfold = F.unfold(x1, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H,
                                                                               W)  # (B, C*n_pts, H, W)
        contents0 = warp(x0_unfold, flow0)
        contents1 = warp(x1_unfold, flow1)

        contents0 = contents0.view(B, C, self.n_pts, H, W)
        contents1 = contents1.view(B, C, self.n_pts, H, W)

        fea0 = contents0[:, :, self.n_pts // 2, :, :]
        fea1 = contents1[:, :, self.n_pts // 2, :, :]

        # get context feature
        occl = self.occl_convs(torch.cat([fea0, fea1], dim=1))
        fea = fea0 * occl + fea1 * (1 - occl)

        # get correlation features
        fea_view = fea.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C)
        contents0 = contents0.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)
        contents1 = contents1.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)

        fea_view = self.L2normalize(fea_view, dim=-1)
        contents0 = self.L2normalize(contents0, dim=-1)
        contents1 = self.L2normalize(contents1, dim=-1)
        corr0 = torch.einsum('bic,bjc->bij', fea_view, contents0)  # (B*H*W, 1, n_pts)
        corr1 = torch.einsum('bic,bjc->bij', fea_view, contents1)
        # corr0 = corr0 / torch.sqrt(torch.tensor(C).float())
        # corr1 = corr1 / torch.sqrt(torch.tensor(C).float())
        corr0 = corr0.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous()  # (B, n_pts, H, W)
        corr1 = corr1.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous()
        corr0 = self.corr_convs(corr0)  # (B, corr_dim, H, W)
        corr1 = self.corr_convs(corr1)

        # get flow features
        flow0_fea = self.flow_convs(flow0)
        flow1_fea = self.flow_convs(flow1)

        # merge correlation and flow features, get motion features
        motion0 = self.motion_convs(torch.cat([corr0, flow0_fea], dim=1))
        motion1 = self.motion_convs(torch.cat([corr1, flow1_fea], dim=1))

        # update flows
        inp0 = torch.cat([fea, fea0, motion0, flow0], dim=1)
        delta_flow0 = self.flow_head(self.gru(inp0))
        flow0 = flow0 + delta_flow0
        inp1 = torch.cat([fea, fea1, motion1, flow1], dim=1)
        delta_flow1 = self.flow_head(self.gru(inp1))
        flow1 = flow1 + delta_flow1

        return flow0, flow1

    def forward(self, x0, x1, flow0, flow1):
        for i in range(self.n_iters):
            flow0, flow1 = self.forward_once(x0, x1, flow0, flow1)

        return torch.cat([flow0, flow1], dim=1)

class FlowRefineNet_Multis(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super(FlowRefineNet_Multis, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

        self.rf_block1 = FlowRefineNetA(context_dim=c, c=c, r=1, n_iters=n_iters)
        self.rf_block2 = FlowRefineNetA(context_dim=2 * c, c=2 * c, r=1, n_iters=n_iters)
        self.rf_block3 = FlowRefineNetA(context_dim=4 * c, c=4 * c, r=1, n_iters=n_iters)
        self.rf_block4 = FlowRefineNetA(context_dim=8 * c, c=8 * c, r=1, n_iters=n_iters)

    def get_context(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def forward(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        # update flow from small scale
        flow = F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25  # 1/8
        flow = self.rf_block4(s_4[:bs], s_4[bs:], flow[:, :2], flow[:, 2:4])  # 1/8
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block3(s_3[:bs], s_3[bs:], flow[:, :2], flow[:, 2:4])  # 1/4
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block2(s_2[:bs], s_2[bs:], flow[:, :2], flow[:, 2:4])  # 1/2
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block1(s_1[:bs], s_1[bs:], flow[:, :2], flow[:, 2:4])  # 1

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs


class FlowRefineNet_Multis_Simple(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super(FlowRefineNet_Multis_Simple, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8
        
        # flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        # print(f'flow_interpolate.shape {flow.shape}')
        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs
