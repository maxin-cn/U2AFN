import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def patchgan(self, outputs, is_real=None, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

    def __call__(self, outputs, is_real=None, is_disc=None):
        return self.patchgan(outputs, is_real, is_disc)

class VGG19(torch.nn.Module):
    def __init__(self, resize_input=False):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.resize_input = resize_input
        self.mean=torch.Tensor([0.485, 0.456, 0.406])
        self.std=torch.Tensor([0.229, 0.224, 0.225])
        prefix = [1,1, 2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5]
        posfix = [1,2, 1,2, 1,2,3,4, 1,2,3,4, 1,2,3,4]
        names = list(zip(prefix, posfix))
        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(pre, pos), torch.nn.Sequential())

        nums = [[0,1], [2,3], [4,5,6], [7,8],
            [9,10,11], [12,13], [14,15], [16,17],
            [18,19,20], [21,22], [23,24], [25,26],
            [27,28,29], [30,31], [32,33], [34,35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # resize and normalize input for pretrained vgg19
        x = (x+1)/2 # [-1, 1] -> [0, 1]
        x = (x-self.mean.view(1,3,1,1).to(x.device)) / (self.std.view(1,3,1,1).to(x.device))
        if self.resize_input:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)
        out = {key: value for (key,value) in list(zip(self.relus, features))}
        return out


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x_vgg, y_vgg):
        # Compute features
        content_loss = 0.0
        prefix = [1,2,3,4,5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(x_vgg['relu{}_1'.format(prefix[i])], 
                                                             y_vgg['relu{}_1'.format(prefix[i])])
        return content_loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x_vgg, y_vgg):
        # Compute loss
        style_loss = 0.0
        prefix = [2,3,4,5]
        posfix = [2,4,4,2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(self.compute_gram(x_vgg['relu{}_{}'.format(pre,pos)]),
                                         self.compute_gram(y_vgg['relu{}_{}'.format(pre, pos)]))
        return style_loss


class TotalVariationLoss(nn.Module):
    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img

        kernel = torch.FloatTensor([
            [0, 1, 0],
            [1, -2, 0],
            [0, 0, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.c_img)

    def forward(self, results, mask):
        loss = 0.
        grad = self.gradient(results) * resize_like(mask, results)
        loss += torch.mean(torch.abs(grad))
        return loss

def resize_like(x, target, mode="bilinear"):
    return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)

# modified from WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, Lambda, masks=None):
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    cuda = torch.cuda.is_available()
    if cuda:
        alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    if cuda:
        # interpolates = interpolates.cuda()
        interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    if masks is not None:
        disc_interpolates = netD(interpolates, masks)
    else:
        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) if cuda else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()
