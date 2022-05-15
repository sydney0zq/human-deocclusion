import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import utils
from . import backbone, InpaintingLoss, AdversarialLoss

def new_gram(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat_t, feat) / (ch * ch)
    return gram

def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

import numpy as np
def th2np(th, dtype='image', transpose=False, rgb_cyclic=False):
    assert dtype in ['image', 'mask']
    if dtype == 'image':
        th = (th + 1.0) / 2.0
        th = th * 255
        npdata = th.detach().cpu().numpy()      # NCHW
        if rgb_cyclic:
            npdata = npdata[:, ::-1, :, :]
        if transpose:
            npdata = npdata.transpose((0, 2, 3, 1)) # NHWC
    else:
        if th.ndim == 3:
            th = th.unsqueeze(1)
        if th.size(1) == 1:
            npdata = th.detach().cpu().repeat(1, 3, 1, 1).numpy()   # NCHW
        else:
            npdata = th.detach().cpu().numpy()
        if transpose:
            npdata = npdata.transpose((0, 2, 3, 1))
    return npdata




class PartialCompletionContentCGAN(nn.Module):

    def __init__(self, params, load_pretrain=None, dist_model=False, is_test=False):
        super(PartialCompletionContentCGAN, self).__init__()
        self.params = params
        self.with_modal = params.get('with_modal', False)

        # model
        self.model = backbone.__dict__[params['backbone_arch']](**params['backbone_param'])
        if load_pretrain is not None:
            assert load_pretrain.endswith('.pth'), "load_pretrain should end with .pth"
            utils.load_weights(load_pretrain, self.model)

        self.model.cuda()

        if dist_model:
            self.model = utils.DistModule(self.model)
            self.world_size = dist.get_world_size()
        else:
            self.model = backbone.FixModule(self.model)
            self.world_size = 1

        self.is_test = is_test
        if is_test:
            return

        # optim
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=params['lr'])

        # netD
        self.netD = backbone.__dict__[params['discriminator']](**params['discriminator_params'])
        self.netD.cuda()
        if dist_model:
            self.netD = utils.DistModule(self.netD)
        else:
            self.netD = backbone.FixModule(self.netD)
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=params['lr'] * params['d2g_lr'], betas=(0.0, 0.9))

        # loss
        self.vgg = backbone.VGG16FeatureExtractor().cuda()
        self.criterion = InpaintingLoss(self.vgg).cuda()
        self.gan_loss = AdversarialLoss(type='nsgan').cuda()

        cudnn.benchmark = True
        self.l1_loss = nn.L1Loss()

    def set_input(self, rgb_erased, modal_mask, amodal_mask, parsing_modal_ohmask, parsing_amodal_ohmask, visible_mask=None, invisible_mask=None, rgb_gt=None):
        def tocuda(x):
            if x is not None:
                return x.cuda()
            else:
                return x
        self.rgb_erased = tocuda(rgb_erased)
        self.modal_mask = tocuda(modal_mask)
        self.amodal_mask = tocuda(amodal_mask)
        self.parsing_modal_ohmask = torch.clamp(tocuda(parsing_modal_ohmask), 0, 1)[:, 1:]
        self.parsing_amodal_ohmask = torch.clamp(tocuda(parsing_amodal_ohmask), 0, 1)[:, 1:]
        self.visible_mask = tocuda(visible_mask)
        self.invisible_mask = tocuda(invisible_mask)
        self.rgb_gt = tocuda(rgb_gt)

        self.raw_rgb_gt = self.rgb_gt.clone()
        self.raw_rgb_erased = self.rgb_erased.clone()

        w = 0.3
        self.rgb_erased = self.rgb_erased * self.amodal_mask + self.rgb_erased * (1-self.amodal_mask) * w
        self.rgb_gt = self.rgb_gt * self.amodal_mask + self.rgb_gt * (1-self.amodal_mask) * w

        self.invparsing_ohmask = torch.clamp(self.parsing_amodal_ohmask - self.parsing_modal_ohmask, 0, 1)


    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            res_dict = self.model(self.rgb_erased, self.visible_mask, self.amodal_mask, self.parsing_modal_ohmask, self.parsing_amodal_ohmask)

        recon_img = res_dict['recon_img']
        comp_img = res_dict['comp_img']

        tb_visual = [self.rgb_erased, self.invisible_mask, recon_img, comp_img, self.rgb_gt]      # NCHW
        is_mask = [False, True, False, False, False, False]
        tb_visual_np = []
        for x, m_bool in zip(tb_visual, is_mask):
            if m_bool is False:
                x = th2np(x, 'image')
            else:
                x = th2np(x*255, 'mask')
            tb_visual_np.append(x)
        
        tb_visual_np = np.concatenate(tb_visual_np, axis=2)[:8].astype(np.uint8)

        recon_img = recon_img * self.amodal_mask + self.raw_rgb_gt * (1-self.amodal_mask)
        comp_img = comp_img * self.amodal_mask + self.raw_rgb_gt * (1-self.amodal_mask)
        
        return {'common_tensors': [self.rgb_erased, recon_img, comp_img, self.rgb_gt],
                'mask_tensors': [self.invisible_mask, self.amodal_mask, self.visible_mask],
                'tb_visual': tb_visual_np,
                'recon_img': recon_img,
                'comp_img': comp_img}


    def step(self):
        res_dict = self.model(self.rgb_erased, self.visible_mask, self.amodal_mask, self.parsing_modal_ohmask, self.parsing_amodal_ohmask)

        recon_img = res_dict['recon_img']
        comp_img = res_dict['comp_img']

        rec_loss = self.l1_loss(recon_img, self.rgb_gt) + self.l1_loss(comp_img, self.rgb_gt) * 5
        
        image_feat = self.vgg(self.rgb_gt)
        recon_feat = self.vgg(recon_img)
        comp_feat = self.vgg(comp_img)
        cor_loss = self.l1_loss(new_gram(recon_feat[1]), new_gram(image_feat[1])) + self.l1_loss(new_gram(comp_feat[1]), new_gram(image_feat[1])) * 5

        style_loss = 0
        for i in range(len(image_feat)):
            style_loss += self.l1_loss(gram_matrix(recon_feat[i]), gram_matrix(image_feat[i]))
            style_loss += self.l1_loss(gram_matrix(comp_feat[i]), gram_matrix(image_feat[i])) * 5

        inpainting_loss = rec_loss * 2 + cor_loss * 1 + style_loss * 40

        d_pred = self.netD(comp_img*self.amodal_mask, self.invisible_mask)
        d_pred_fake = self.netD(comp_img.detach()*self.amodal_mask, self.invisible_mask)
        d_real = self.netD(self.rgb_gt * self.amodal_mask, self.invisible_mask)

        adv_d_loss_real = 0.01 * self.gan_loss(d_real, 1)
        adv_d_loss_fake = self.gan_loss(d_pred_fake, 0)
        adv_d_loss = adv_d_loss_real + adv_d_loss_fake
        adv_g_loss = self.gan_loss(d_pred, 1)

        adv_loss = adv_d_loss + adv_g_loss

        # create loss dict
        loss_dict = {}
        loss_dict['dis'] = adv_d_loss
        loss_dict['gen'] = adv_g_loss
        loss_dict['inpainting'] = inpainting_loss

        # update
        self.optimD.zero_grad()
        adv_loss.backward(retain_graph=True)
        utils.average_gradients(self.netD)
        self.optimD.step()

        self.optim.zero_grad()
        inpainting_loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()

        return loss_dict

    def load_model_demo(self, path):
        utils.load_state(path, self.model)

    def load_state(self, root, Iter, resume=False):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))
        print ("loading G path: {}".format(path))

        if resume:
            utils.load_state(path, self.model, self.optim)
            if not self.is_test:
                utils.load_state(netD_path, self.netD, self.optimD)
        else:
            utils.load_state(path, self.model)
            if not self.is_test:
                utils.load_state(netD_path, self.netD)

    def save_state(self, root, Iter):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

        torch.save({
            'step': Iter,
            'state_dict': self.netD.state_dict(),
            'optimizer': self.optimD.state_dict()}, netD_path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
            self.netD.train()
        else:
            self.model.eval()
            if not self.is_test:
                self.netD.eval()
