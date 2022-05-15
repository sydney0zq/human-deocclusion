# full assembly of the sub-parts to form the complete net
import torch.nn as nn
import torch.nn.functional as F
import torch

import os, sys
CURR_DIR = os.path.dirname(__file__)

class ReweightAnchorLogic(nn.Module):
    def __init__(self, num_anchors=64, k=16, se_weight=0.5):
        super(ReweightAnchorLogic, self).__init__()
        self.k = k
        self.anchor_conv = nn.Conv2d(num_anchors, num_anchors//2, 3, 1, 1)

    def forward(self, x, anchors):
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        anchors = F.interpolate(anchors.unsqueeze(0), (h, w), mode="nearest")
        anchors = anchors.repeat(batch_size, 1, 1, 1)
        
        affinity = torch.sum(x*anchors, dim=[2, 3])    # NxM

        # keep all anchors, only reweight them
        affinity = F.softmax(affinity, dim=1)
        reweight_anchors = anchors * affinity.view(batch_size, -1, 1, 1)      # NxMxHxW

        # keep only the top k and reweight them
        # topk_affinity, topk_indices = torch.topk(affinity, self.k, dim=1, largest=True)

        # topk_anchors = []
        # for i in range(batch_size):
        #     topk_anchors.append(anchors[i:i+1, topk_indices[i]])
        # topk_anchors = torch.cat(topk_anchors, dim=0)
        # topk_affinity = F.softmax(topk_affinity)
        # reweight_anchors = topk_anchors * topk_affinity.view(batch_size, -1, 1, 1)      # NxMxHxW
        
        reweight_anchors = self.anchor_conv(reweight_anchors)
        return reweight_anchors




class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x + r

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


################################################################################


class UNetD2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNetD2, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 32 * w)
        self.up1 = up(64 * w, 16 * w)
        self.up2 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

    def forward(self, x):
        x1 = self.inc(x) # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 32
        x = self.up1(x3, x2) # 16
        x = self.up2(x, x1) # 16
        x = self.outc(x)
        return x


class UNetD3(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNetD3, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 64 * w)
        self.down3 = down(64 * w, 64 * w)
        self.up2 = up(128 * w, 32 * w)
        self.up3 = up(64 * w, 16 * w)
        self.up4 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet_base(nn.Module):
    def __init__(self, in_channels=4, w=4, out_channels=2, parsing_channels=20):
        super(UNet_base, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))

        self.modal_conv = nn.Sequential(
                nn.Conv2d(int(16*w)+32, 16, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
                )
        self.parsing_modal_conv_pre = nn.Sequential(
                ResBlock(int(16*w), 32),
                nn.ReLU(inplace=True),
                )
        self.se_modal = SELayer(channel=32, reduction=4)
        self.parsing_modal_conv = nn.Conv2d(32, parsing_channels, kernel_size=3, stride=1, padding=1)

        self.inc_hg2 = inconv(int(16 * w)+3, int(16 * w))
        self.down1_hg2 = down(int(16 * w), int(32 * w))
        self.down2_hg2 = down(int(32 * w), int(64 * w))
        self.down3_hg2 = down(int(64 * w), int(128 * w))
        self.down4_hg2 = down(int(128 * w), int(128 * w))
        self.up1_hg2 = up(int(256 * w), int(64 * w))
        self.up2_hg2 = up(int(128 * w), int(32 * w))
        self.up3_hg2 = up(int(64 * w), int(16 * w))
        self.up4_hg2 = up(int(32 * w), int(16 * w))

        self.amodal_conv = nn.Sequential(
                nn.Conv2d(int(16*w)+32+32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
                )
        self.parsing_amodal_conv_pre = nn.Sequential(
                ResBlock(int(16*w), 32),
                nn.ReLU(inplace=True),
                )
        self.se_amodal = SELayer(channel=32, reduction=4)
        self.parsing_amodal_conv = nn.Conv2d(32, parsing_channels, kernel_size=3, stride=1, padding=1)

        self.register_buffer('anchors', torch.load(os.path.join(CURR_DIR, "centers/centers_64.pth")).float())
        # self.deform_conv_layer = DeformConv2dPack(64, 64, 3, 1, 1)

        self.reweight_gate = ReweightAnchorLogic()


    def forward(self, x):
        image = x[:, :3]
        n, c, h, w = x.size()

        # The first HG outputs modal and parsing_modal_target
        x1 = self.inc(x)    # 1
        x2 = self.down1(x1) # 1/2
        x3 = self.down2(x2) # 1/4
        x4 = self.down3(x3) # 1/8
        x5 = self.down4(x4) # 1/16
        x54 = self.up1(x5, x4)  # 1/8
        x43 = self.up2(x54, x3) # 1/4
        x32 = self.up3(x43, x2)
        x21 = self.up4(x32, x1)

        m_parsing_feat = self.parsing_modal_conv_pre(x21)
        parsing_modal_logit = self.parsing_modal_conv(m_parsing_feat)

        m_parsing_aux_feat = self.se_modal(m_parsing_feat.detach())
        modal_feat = torch.cat([x21, m_parsing_aux_feat], dim=1)
        modal_logit = self.modal_conv(modal_feat)

        modal_fg = F.softmax(modal_logit, dim=1)[:, 1:2]
        corr_anchors = self.reweight_gate(modal_fg.detach(), self.anchors)
        
        hg2_infeat = torch.cat([image, x21], dim=1)
        x1_hgp = self.inc_hg2(hg2_infeat)
        x2_hgp = self.down1_hg2(x1_hgp)
        x3_hgp = self.down2_hg2(x2_hgp) # 1/4
        x4_hgp = self.down3_hg2(x3_hgp) # 1/8
        x5_hgp = self.down4_hg2(x4_hgp) # 1/16
        x54_hgp = self.up1_hg2(x5_hgp, x4_hgp)  # 1/8
        x43_hgp = self.up2_hg2(x54_hgp, x3_hgp) # 1/4
        x32_hgp = self.up3_hg2(x43_hgp, x2_hgp)
        x21_hgp = self.up4_hg2(x32_hgp, x1_hgp)

        am_parsing_feat = self.parsing_amodal_conv_pre(x21_hgp)
        parsing_amodal_logit = self.parsing_amodal_conv(am_parsing_feat)

        am_parsing_aux_feat = self.se_amodal(am_parsing_feat.detach())
        amodal_feat = torch.cat([x21_hgp, am_parsing_aux_feat, corr_anchors], dim=1)
        amodal_logit = self.amodal_conv(amodal_feat)

        return {'modal': modal_logit, 'amodal': amodal_logit, 
                'parsing_modal': parsing_modal_logit, 
                'parsing_amodal': parsing_amodal_logit}

def unet(in_channels, **kwargs):
    return UNet_base(**kwargs)


if __name__ == "__main__":
    size = (1, 4, 256, 256)
    device = torch.device('cuda')
    model = unet(4).to(device)
    import numpy as np
    dummy_input = torch.randn(size, dtype=torch.float).to(device)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            if rep % 10 == 0:
                print (rep)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)

