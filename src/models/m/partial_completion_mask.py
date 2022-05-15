import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss, DiceLoss, lovasz_softmax

from . import backbone
from .detection.detection_api import Detector

import os, sys
CURR_DIR = os.path.dirname(__file__)


class PartialCompletionMask(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(PartialCompletionMask, self).__init__(params, load_pretrain, dist_model)

        self.params = params
        
        try:
            self.criterion = MaskWeightedCrossEntropyLoss(
                                inmask_weight=params['inmask_weight'],
                                outmask_weight=params['outmask_weight'])
        except:
            print ("=> Ignoring criterion...")
        self.celoss = nn.CrossEntropyLoss()
        self.iouloss_1 = DiceLoss()
        self.iouloss_2 = lovasz_softmax
        self.detector = Detector(config_file=os.path.join(CURR_DIR, "./detection/maskrcnn/configs/centermask-V2-99-FPN-ms-3x.yaml"), 
                    model_path=os.path.join(CURR_DIR, "./detection/pretrained/centermask-V2-99-FPN-ms-3x.pth"))
        self.l1 = nn.L1Loss()

        self.extractor = backbone.VGG16FeatureExtractor().cuda()
        self.extractor.cuda()

        for param in self.extractor.parameters():
            param.requires_grad = False
        
    def set_input(self, image, image4det, modal_target, amodal_target, parsing_modal_target=None, parsing_amodal_target=None, ret_info=None):
        self.image = image.cuda()
        self.detmodal = self.detector.infer_image_th(image4det)
        self.modal_target = modal_target.cuda()
        self.amodal_target = amodal_target.cuda()
        self.invmask_target = torch.clamp(amodal_target-modal_target, 0, 1).cuda()
        if parsing_modal_target is None:
            self.parsing_modal_target = None
            self.parsing_amodal_target = None
        else:
            self.parsing_modal_target = parsing_modal_target.cuda()
            self.parsing_amodal_target = parsing_amodal_target.cuda()
        self.ret_info = ret_info

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            res_dict = self.model(torch.cat([self.image, self.detmodal], dim=1))
        
        out_modal = res_dict['modal'].argmax(dim=1, keepdim=False)
        out_amodal = res_dict['amodal'].argmax(dim=1, keepdim=False)
        out_amodal = out_amodal | out_modal
        out_modal_parsing = res_dict['parsing_modal'].argmax(dim=1, keepdim=False)
        out_amodal_parsing = res_dict['parsing_amodal'].argmax(dim=1, keepdim=False)
        out_modal_parsing = out_modal_parsing * out_modal
        out_amodal_parsing = out_amodal_parsing * out_amodal

        if self.parsing_modal_target is not None:
            ret_tensors = {'visual_tensors': [self.image,
                                              self.detmodal.repeat(1, 3, 1, 1),
                                              self.modal_target.unsqueeze(1).repeat(1, 3, 1, 1), 
                                              self.amodal_target.unsqueeze(1).repeat(1, 3, 1, 1),
                                              out_modal.unsqueeze(1).repeat(1, 3, 1, 1), 
                                              out_amodal.unsqueeze(1).repeat(1, 3, 1, 1),
                                              self.parsing_modal_target.unsqueeze(1).repeat(1, 3, 1, 1)*12,
                                              out_modal_parsing.unsqueeze(1).repeat(1, 3, 1, 1)*12,
                                              self.parsing_amodal_target.unsqueeze(1).repeat(1, 3, 1, 1)*12,
                                              out_amodal_parsing.unsqueeze(1).repeat(1, 3, 1, 1)*12,
                                              ],
                                              
                        #  'mask_tensors': [self.modal_mask, vis_combo, vis_target],
                           'mask_bool': [False, True, True, True, True, True, True, True, True, True],
                           'prediction': [out_modal, out_amodal],
                           'target': [self.modal_target, self.amodal_target]}
        else:
            ret_tensors = {'visual_tensors': [self.image,
                                              self.detmodal.repeat(1, 3, 1, 1),
                                              self.modal_target.unsqueeze(1).repeat(1, 3, 1, 1), 
                                              self.amodal_target.unsqueeze(1).repeat(1, 3, 1, 1),
                                              out_modal.unsqueeze(1).repeat(1, 3, 1, 1), 
                                              out_amodal.unsqueeze(1).repeat(1, 3, 1, 1),
                                              ],
                                              
                        #  'mask_tensors': [self.modal_mask, vis_combo, vis_target],
                           'mask_bool': [False, True, True, True, True, True],
                           'prediction': [out_modal, out_amodal],
                           'target': [self.modal_target, self.amodal_target]}


        if ret_loss:
            #loss = self.criterion(output, self.amodal_target, self.modal_mask.squeeze(1)) /  self.world_size
            return ret_tensors, {'loss': -1}
        else:
            return ret_tensors


    def step(self):
        res_dict = self.model(torch.cat([self.image, self.detmodal], dim=1))

        # compute modal loss
        loss_modal = self.celoss(res_dict['modal'], self.modal_target)
        loss_parsing_modal = self.celoss(res_dict['parsing_modal'], self.parsing_modal_target)

        loss_amodal = self.celoss(res_dict['amodal'], self.amodal_target)
        loss_parsing_amodal = self.celoss(res_dict['parsing_amodal'], self.parsing_amodal_target)

        ce_loss = (loss_modal + loss_amodal + loss_parsing_modal * 0.2 + loss_parsing_amodal * 0.2) / 2.0

        out_modal, out_amodal = res_dict['modal'], res_dict['amodal']
        out_modal = out_modal.argmax(dim=1, keepdim=True)
        out_amodal = out_amodal.argmax(dim=1, keepdim=True)

        out_amodal_bin = out_amodal.float()
        out_modal_bin = out_modal.float()
        target_amodal = self.amodal_target.unsqueeze(1).float()
        target_modal = self.modal_target.unsqueeze(1).float()
        l1_loss = self.l1(out_amodal_bin, target_amodal) + self.l1(out_modal_bin, target_modal)
        
        target_feat = self.extractor(target_amodal.repeat(1, 3, 1, 1))
        pred_feat = self.extractor(out_amodal_bin.repeat(1, 3, 1, 1))

        prc_loss = 0
        for i in range(3):
            prc_loss += self.l1(target_feat[i], pred_feat[i])
        loss = ce_loss + (l1_loss + prc_loss)*0.1

        # loss = self.celoss(out_amodal, self.amodal_target)
        #out_amodal_p = F.softmax(out_amodal, dim=1)
        #loss_amodal += self.iouloss_1(out_amodal_p[:, 1], self.amodal_target) + self.iouloss_2(out_amodal_p, self.amodal_target)

        #loss = (loss_modal + 10*loss_invmask + loss_amodal)/2.0
        #loss = (loss_modal + loss_invmask)/2.0
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()

        # optimize discriminator
        self.optim_D.zero_grad()
        d_real = self.netD(self.image, self.amodal_target.unsqueeze(1).float())
        d_fake = self.netD(self.image, out_amodal_bin)
        adv_d_loss_real = self.gan_loss(d_real, 1)
        adv_d_loss_fake = self.gan_loss(d_fake, 0)
        adv_d_loss = adv_d_loss_fake + adv_d_loss_real
        adv_d_loss.backward()
        self.optim_D.step()

        return {'loss': loss, 'd_loss': adv_d_loss}
