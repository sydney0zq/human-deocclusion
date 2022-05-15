import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import backbone
import utils

from .dis import Discriminator, AdversarialLoss

class SingleStageModel(object):

    def __init__(self, params, load_path=None, dist_model=False): # set dist_model to False in demo
        self.model = backbone.__dict__[params['backbone_arch']](**params['backbone_param'])
        # Random init here, so do not load pretrained model when constructing backbones.
        utils.init_weights(self.model, init_type='xavier')
        # load pretrain here.
        if load_path is not None:
            utils.load_weights(load_path, self.model)
        self.model.cuda()
        self.netD = Discriminator(in_channels=4)
        self.netD.cuda()
        self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=0.001, betas=(0.5, 0.9))
        self.gan_loss = AdversarialLoss(type='nsgan').cuda()

        if dist_model:
            self.model = utils.DistModule(self.model)
            self.netD = utils.DistModule(self.netD)
            self.world_size = dist.get_world_size()
        else:
            self.model = backbone.FixModule(self.model)
            self.netD = backbone.FixModule(self.netD)
            self.world_size = 1
        
        self.ckpt_templ = params['ckpt_templ'] 

        try:
            if params['optim'] == 'SGD':
                self.optim = torch.optim.SGD(
                    self.model.parameters(), lr=params['lr'],
                    momentum=0.9, weight_decay=params['weight_decay'])
            elif params['optim'] == 'Adam':
                self.optim = torch.optim.Adam(
                    self.model.parameters(), lr=params['lr'],
                    betas=(params['beta1'], 0.999))
            else:   
                print("No optimzier")
        except:
            pass

        cudnn.benchmark = True

    def forward(self, ret_loss=True):
        pass

    def step(self):
        pass

    def load_state(self, path, Iter, resume=False):
        path = os.path.join(path, self.ckpt_templ.format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
            
            self.netD.load_state_dict(torch.load(path.replace('.pth', '_D.pth')))
        else:
            utils.load_state(path, self.model)
        

    def save_state(self, path, Iter):
        path = os.path.join(path, self.ckpt_templ.format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)
        print ('saving D: {}'.format(path.replace('.pth', '_D.pth')))
        torch.save(self.netD.state_dict(), path.replace('.pth', '_D.pth'))

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
