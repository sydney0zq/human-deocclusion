import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = ['PConvUNet', 'VGG16FeatureExtractor']

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activ == 'elu':
            self.activation = nn.ELU()

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


# 2020-01-07: SE Attention layer
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


class ParsingFuseAttention(nn.Module):
    def __init__(self, in_dim):
        super(ParsingFuseAttention, self).__init__()
        self.in_dim = in_dim
        self.modal_parsing_conv = nn.Conv2d(in_dim, 19, 3, 1, 1)
        self.amodal_parsing_conv = nn.Conv2d(in_dim, 19, 3, 1, 1)
        self.selayer = SELayer(in_dim, reduction=8)

        self.q_conv = nn.Conv2d(in_dim+19, in_dim//8, 3, 1, 1)
        self.k_conv = nn.Conv2d(in_dim+19, in_dim//8, 3, 1, 1)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1)

        self.comp_conv = nn.Conv2d(in_dim+19+19, in_dim, 3, 1, 1)
        self.out_conv = nn.Conv2d(in_dim*3, in_dim, 3, 1, 1)

    def forward(self, x, modalparsing_ohmask, amodalparsing_ohmask):
        b, c, h, w = x.size()
        modalparsing_ohmask = F.interpolate(modalparsing_ohmask, size=(h, w), mode="nearest")
        amodalparsing_ohmask = F.interpolate(amodalparsing_ohmask, size=(h, w), mode="nearest")
        # dyvis_mask = F.interpolate(dyvis_mask, size=(h, w), mode="nearest")
        # dyinvis_mask = 1-dyvis_mask

        modal_x = self.modal_parsing_conv(x) * modalparsing_ohmask
        amodal_x = self.amodal_parsing_conv(x) * amodalparsing_ohmask

        # Get to know which channel should be focused
        input_x = torch.cat([x, modal_x, amodal_x], dim=1)
        out = self.comp_conv(input_x)
        out_a = self.selayer(out)

        # attention based on the parsing oh masks
        q_x = self.q_conv(torch.cat([x, modalparsing_ohmask], dim=1)).view(b,-1,w*h).permute(0,2,1)     # BxCxN -> BxNxC
        k_x = self.k_conv(torch.cat([x, amodalparsing_ohmask], dim=1)).view(b,-1,w*h)                    # BxCxN
        v_x = self.v_conv(x).view(b,-1,w*h)                    # BxCxN

        energy = torch.bmm(q_x, k_x) # transpose check  # BxNxN
        attention = F.softmax(energy, dim=-1) # BX (N) X (N)
        attention_t = F.softmax(energy.permute(0, 2, 1), dim=-1)
        out_b = torch.bmm(v_x, attention.permute(0, 2, 1))
        out_c = torch.bmm(v_x, attention_t.permute(0, 2, 1))
        out_b = out_b.view(b, c, h, w)
        out_c = out_c.view(b, c, h, w)

        out = torch.cat([out_a, out_b, out_c], dim=1)
        out = self.out_conv(out)
        return out


class ParsingFuseFeatureModule(nn.Module):
    def __init__(self, in_dim):
        super(ParsingFuseFeatureModule, self).__init__()
        self.modal_parsing_conv = nn.Conv2d(in_dim, 19, 3, 1, 1)
        self.amodal_parsing_conv = nn.Conv2d(in_dim, 19, 3, 1, 1)

        self.comp_conv = nn.Conv2d(in_dim+19+19, in_dim, 3, 1, 1)

    def forward(self, x, modalparsing_ohmask, amodalparsing_ohmask):
        n, c, h, w = x.size()
        modalparsing_ohmask = F.interpolate(modalparsing_ohmask, size=(h, w), mode="nearest")
        amodalparsing_ohmask = F.interpolate(amodalparsing_ohmask, size=(h, w), mode="nearest")

        modal_x = self.modal_parsing_conv(x) * modalparsing_ohmask
        amodal_x = self.amodal_parsing_conv(x) * amodalparsing_ohmask

        input_x = torch.cat([x, modal_x, amodal_x], dim=1)
        out = self.comp_conv(input_x)
        return out

class DynamicVisible2InvisibleAttention(nn.Module):
    def __init__(self, in_dim):
        super(DynamicVisible2InvisibleAttention, self).__init__()
        self.in_dim = in_dim
        self.q_conv = nn.Conv2d(in_dim, in_dim//8, 3, 1, 1)
        self.k_conv = nn.Conv2d(in_dim, in_dim//8, 3, 1, 1)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.dynamic_conv = nn.Conv2d(in_dim, 1, 3, 1, 1, bias=False)
        self.out_conv = nn.Conv2d(in_dim*2, in_dim, 3, 1, 1, bias=False)

    # x is in the shape of NCHW
    # dymodal_mask has the same channel size with x
    def forward(self, x, dyvis_mask):
        b, c, h, w = x.size()
        # print (x.size(), self.in_dim)
        q_x = self.q_conv(x).view(b,-1,w*h).permute(0,2,1)     # BxCxN -> BxNxC
        k_x = self.k_conv(x).view(b,-1,w*h)                    # BxCxN
        v_x = self.v_conv(x).view(b,-1,w*h)                    # BxCxN

        dyvis_mask = torch.clamp(self.dynamic_conv(dyvis_mask), 0, 1)
        dyinvis_mask = 1-dyvis_mask

        mm_resampled = F.interpolate(dyvis_mask, size=(h, w), mode="nearest")
        im_resmapled = F.interpolate(dyinvis_mask, size=(h, w), mode="nearest")
        mm_resampled_col = mm_resampled.view(b, 1, w*h)
        im_resampled_row = im_resmapled.view(b, w*h, 1)

        energy = torch.bmm(q_x, k_x) # transpose check  # BxNxN

        energy = energy * mm_resampled_col * im_resampled_row

        # multiply pm onto the energy
        attention = F.softmax(energy, dim=-1) # BX (N) X (N)
        out = torch.bmm(v_x, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = torch.cat([x, out], dim=1)
        out = self.out_conv(out)

        return out




def th_upsample2x(x, mode='nearest'):
    x = F.interpolate(x, scale_factor=2, mode=mode)
    return x

class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3):
        super().__init__()
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_3_att = ParsingFuseAttention(256)
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_5_att = ParsingFuseAttention(512)
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5_att = ParsingFuseAttention(512)
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_3_att = ParsingFuseAttention(128)
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, 3,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, rgb_erased, visible_mask, amodal_mask, modalparsing_ohmask, amodalparsing_ohmask):
        h0 = torch.cat([rgb_erased, visible_mask, amodal_mask], dim=1)
        h0_mask = visible_mask.repeat(1, h0.size(1), 1, 1)
        h1, h1_mask = self.enc_1(h0, h0_mask)
        h2, h2_mask = self.enc_2(h1, h1_mask)
        h3, h3_mask = self.enc_3(h2, h2_mask)
        h3 = self.enc_3_att(h3, modalparsing_ohmask, amodalparsing_ohmask)
        h4, h4_mask = self.enc_4(h3, h3_mask)
        h5, h5_mask = self.enc_5(h4, h4_mask)
        h5 = self.enc_5_att(h5, modalparsing_ohmask, amodalparsing_ohmask)
        h6, h6_mask = self.enc_6(h5, h5_mask)
        h7, h7_mask = self.enc_7(h6, h6_mask)

        h7d, h7d_mask = th_upsample2x(h7), th_upsample2x(h7_mask)
        h67, h67_mask = torch.cat([h7d, h6], dim=1), torch.cat([h7d_mask, h6_mask], dim=1)
        h6d, h6d_mask = self.dec_7(h67, h67_mask)

        h6d, h6d_mask = th_upsample2x(h6d), th_upsample2x(h6d_mask)
        h56, h56_mask = torch.cat([h6d, h5], dim=1), torch.cat([h6d_mask, h5_mask], dim=1)
        h5d, h5d_mask = self.dec_6(h56, h56_mask)

        h5d, h5d_mask = th_upsample2x(h5d), th_upsample2x(h5d_mask)
        h45, h45_mask = torch.cat([h5d, h4], dim=1), torch.cat([h5d_mask, h4_mask], dim=1)
        h4d, h4d_mask = self.dec_5(h45, h45_mask)
        h4d = self.dec_5_att(h4d, modalparsing_ohmask, amodalparsing_ohmask)

        h4d, h4d_mask = th_upsample2x(h4d), th_upsample2x(h4d_mask)
        h34, h34_mask = torch.cat([h4d, h3], dim=1), torch.cat([h4d_mask, h3_mask], dim=1)
        h3d, h3d_mask = self.dec_4(h34, h34_mask)

        h3d, h3d_mask = th_upsample2x(h3d), th_upsample2x(h3d_mask)
        h23, h23_mask = torch.cat([h3d, h2], dim=1), torch.cat([h3d_mask, h2_mask], dim=1)
        h2d, h2d_mask = self.dec_3(h23, h23_mask)
        h2d = self.dec_3_att(h2d, modalparsing_ohmask, amodalparsing_ohmask)

        h2d, h2d_mask = th_upsample2x(h2d), th_upsample2x(h2d_mask)
        h12, h12_mask = torch.cat([h2d, h1], dim=1), torch.cat([h2d_mask, h1_mask], dim=1)
        h1d, h1d_mask = self.dec_2(h12, h12_mask)

        h1d, h1d_mask = th_upsample2x(h1d), th_upsample2x(h1d_mask)
        h01, h01_mask = torch.cat([h1d, h0], dim=1), torch.cat([h1d_mask, h0_mask], dim=1)
        h0d, h0d_mask = self.dec_1(h01, h01_mask)
        recon_img = torch.clamp(h0d, -1, 1)

        comp_img = rgb_erased * visible_mask + recon_img * (1-visible_mask)

        res_dict = {'recon_img': recon_img, 'comp_img': comp_img}
        return res_dict

    def train(self, mode=True):
        """ Override the default train() to freeze the BN parameters """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


if __name__ == "__main__":            
    size = (1, 3, 256, 256)                                                                                                                             
    device = torch.device('cuda')
    model = PConvUNet(7, 5).to(device) 
    import numpy as np                                                      
    dummy_input1 = torch.randn(size, dtype=torch.float).to(device)          
    dummy_input2 = torch.randn((1, 1, 256, 256), dtype=torch.float).to(device)
    dummy_input3 = torch.randn((1, 1, 256, 256), dtype=torch.float).to(device)
    dummy_input4 = torch.randn((1, 19, 256, 256), dtype=torch.float).to(device)
    dummy_input5 = torch.randn((1, 19, 256, 256), dtype=torch.float).to(device)
    dummy_input = [dummy_input1, dummy_input2, dummy_input3, dummy_input4, dummy_input5]

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300                 
    timings=np.zeros((repetitions,1))                                                                                                                   

                                                                            
    #GPU-WARM-UP                                                            
    for _ in range(10):                                                     
        _ = model(*dummy_input)                                             
                                                                                                                                                        
    # MEASURE PERFORMANCE                                                                                                                               
    with torch.no_grad():                                                                                                                               
        for rep in range(repetitions):                                      
            if rep % 10 == 0:         
                print (rep)                                                                                                                             
            starter.record()                                                
            _ = model(*dummy_input)                                                                                                                     
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)         
    print(mean_syn)                                                         
                                      



if __name__ == '__main2__':
    size = (1, 3, 5, 5)
    input = torch.ones(size)
    input_mask = torch.ones(size)
    input_mask[:, :, 2:, :][:, :, :, 2:] = 0

    conv = PartialConv(3, 3, 3, 1, 1)
    l1 = nn.L1Loss()
    input.requires_grad = True

    output, output_mask = conv(input, input_mask)
    loss = l1(output, torch.randn(1, 3, 5, 5))
    loss.backward()

    assert (torch.sum(input.grad != input.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.bias.grad)).item() == 0)

    # model = PConvUNet()
    # output, output_mask = model(input, input_mask)
