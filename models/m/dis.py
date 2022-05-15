import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as n
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCEWithLogitsLoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            if isinstance(outputs[0], list):
                #loss = 0.0
                losses = []
                for output  in outputs:
                    pred = output[-1]
                    target_tensor = self.get_target_tensor(pred, is_real)
                    #loss += self.criterion(pred, target_tensor)
                    loss = self.criterion(pred, target_tensor)
                    losses.append(loss)
                #return loss / float(len(outputs))
                return losses
            else:
                target_tensor = self.get_target_tensor(outputs, is_real)
                loss = self.criterion(outputs, target_tensor)
                return loss



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# 1/8 stride
class Discriminator(nn.Module):
    def __init__(self, in_channels=4, use_spectral_norm=True, kz=5):
        super(Discriminator, self).__init__()

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kz, stride=2, padding=2), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kz, stride=2, padding=2), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kz, stride=2, padding=2), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5