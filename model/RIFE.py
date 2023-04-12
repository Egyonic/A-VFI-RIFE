import os.path

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from model.VFIformer import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, args, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        # transformer部分
        c = 24
        height = 192 # args.crop_size
        width = 192 # args.crop_size
        window_size = 4
        embed_dim = 136
        self.refinenet = FlowRefineNet_Multis_Simple(c=c, n_iters=1)
        self.fuse_block = nn.Sequential(nn.Conv2d(12, 2 * c, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True), )

        self.transformer = TFModel(img_size=(height, width), in_chans=2 * c, out_chans=4, fuse_c=c,
                                   window_size=window_size, img_range=1.,
                                   depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                   embed_dim=embed_dim, num_heads=[[2, 2], [2, 2], [2, 2], [2, 2]], mlp_ratio=2,
                                   resi_connection='1conv',
                                   use_crossattn=[[[False, False, False, False], [True, True, True, True]], \
                                                  [[False, False, False, False], [True, True, True, True]], \
                                                  [[False, False, False, False], [True, True, True, True]], \
                                                  [[False, False, False, False], [False, False, False, False]]])
        self.device()
        g_params = [self.flownet.parameters(), self.refinenet.parameters(), self.fuse_block.parameters(), self.transformer.parameters()]
        self.optimG = AdamW(itertools.chain.from_iterable(g_params), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
            self.refinenet = DDP(self.refinenet, device_ids=[local_rank], output_device=local_rank)
            self.fuse_block = DDP(self.fuse_block, device_ids=[local_rank], output_device=local_rank)
            self.transformer = DDP(self.transformer, device_ids=[local_rank], output_device=local_rank)

        self.criterion_l1 = nn.L1Loss().to(device)

        ## init loss and optimizer
        # if args.phase == 'train':
        #     if args.rank <= 0:
        #         logging.info('init criterion and optimizer...')
            # g_params = [self.flownet.parameters()]

            # self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable(g_params), lr=args.lr, weight_decay=args.weight_decay)
            # self.scheduler = CosineAnnealingLR(self.optimizer_G, T_max=500)  # T_max=args.max_iter
            # self.optimizer_G = torch.optim.AdamW(itertools.chain.from_iterable(g_params), lr=args.lr,
            #                                      weight_decay=args.weight_decay)


            # if args.loss_flow:
            #     self.criterion_flow = EPE().to(self.device)
            #     self.lambda_flow = args.lambda_flow
            #     if args.rank <= 0:
            #         logging.info('  using flow loss...')
            #
            # if args.loss_ter:
            #     self.criterion_ter = Ternary(self.device).to(self.device)
            #     self.lambda_ter = args.lambda_ter
            #     if args.rank <= 0:
            #         logging.info('  using ter loss...')
            #
            # if args.loss_adv:
            #     self.criterion_adv = AdversarialLoss(gpu_ids=args.gpu_ids, dist=args.dist, gan_type=args.gan_type,
            #                                          gan_k=1, lr_dis=args.lr_D, train_crop_size=40)
            #     self.lambda_adv = args.lambda_adv
            #     if args.rank <= 0:
            #         logging.info('  using adv loss...')
            #
            # if args.loss_perceptual:
            #     self.criterion_perceptual = PerceptualLoss(layer_weights={'conv5_4': 1.}).to(self.device)
            #     self.lambda_perceptual = args.lambda_perceptual
            #     if args.rank <= 0:
            #         logging.info('  using perceptual loss...')

            # if args.resume_optim:
            #     self.load_networks('optimizer_G', self.args.resume_optim)
            # if args.resume_scheduler:
            #     self.load_networks('scheduler', self.args.resume_scheduler)
    def train(self):
        self.flownet.train()
        self.refinenet.train()
        self.fuse_block.train()
        self.transformer.train()

    def eval(self):
        self.flownet.eval()
        self.refinenet.eval()
        self.fuse_block.eval()
        self.transformer.eval()

    def device(self):
        self.flownet.to(device)
        self.refinenet.to(device)
        self.fuse_block.to(device)
        self.transformer.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            # self.flownet.load_state_dict(convert(torch.load(os.path.join(path, 'flownet_m.pkl'))))

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))
            torch.save(self.refinenet.state_dict(), '{}/refinenet.pth'.format(path))
            torch.save(self.fuse_block.state_dict(), '{}/fuse_block.pth'.format(path))
            torch.save(self.transformer.state_dict(), '{}/transformer.pth'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        flow, flow_list, flow_teacher, merged, warped_img0_rife, warped_img1_rife, merged_teacher, loss_distill \
            = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])

        print(f'img0: {img0.shape}')
        flow_refine, c0, c1 = self.refinenet(img0, img1, flow)
        print(f'flow: {flow.shape}')
        print(f'flow_refine: {flow_refine.shape}')
        for i in range(len(c0)):
            print(f'c0[{i}] :{c0[i].shape}')
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:])
        #print(c0.shape)
        x = self.fuse_block(torch.cat([img0, img1, warped_img0, warped_img1], dim=1))
        print(f'x.shape: {x.shape}')
        refine_output = self.transformer(x, c0, c1)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        # loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            loss = 0
            self.optimG.zero_grad()
            # loss_G = loss_l1 + loss_tea + loss_distill * 0.01 # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            l1_loss = self.criterion_l1(pred, gt)
            # l1_loss = l1_loss * self.lambda_l1
            loss = l1_loss + loss_distill * 0.01 + loss_tea

            loss.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return pred, {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[0][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': l1_loss,
            'loss_distill': loss_distill
            }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
