import torch
from torch import nn
from torchvision import models
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torchvision.utils as vutils 

from utils.utils import resize_like
from networks.discriminator import Discriminator
from solvers.my_loss import *

class ReconstructionLoss(nn.L1Loss):
    def __init__(self, lambda_hole, lambda_valid):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.lambda_hole = lambda_hole
        self.lambda_valid = lambda_valid

    def forward(self, predicts, targets, masks, uncertainty=None):
        # uncertainty is map instead of scalar/vector
        if uncertainty is None:
            hole_loss = self.l1_loss(predicts*masks*(1/uncertainty), targets*masks*(1/uncertainty)) / torch.mean(masks)
            valid_loss = self.l1_loss(predicts*(1-masks)*(1/uncertainty), targets*(1-masks)*(1/uncertainty)) / torch.mean(1-masks)
        else:
            hole_loss = self.l1_loss(predicts*masks, targets*masks) / torch.mean(masks)
            valid_loss = self.l1_loss(predicts*(1-masks), targets*(1-masks)) / torch.mean(1-masks)

        loss = hole_loss * self.lambda_hole + valid_loss * self.lambda_valid
        return loss

def img2show(gt_imgs, pred_imgs, comp_imgs, inputs, display_number=4, normalize=False):
    img_to_show = torch.cat([gt_imgs, pred_imgs, comp_imgs, inputs], dim=2)[:display_number]
    img_to_show = make_grid(img_to_show, nrow=display_number, normalize=True, scale_each=True)
    return img_to_show

class InpaintLoss(nn.Module):
    def __init__(self, c_img=3, loss_weights={}, gan_type='lsgan', logPath=None, wo_uncertainty=False):
        super().__init__()
        self.loss_weights = loss_weights
        self.gan_type = gan_type
        # self.wo_uncertainty = wo_uncertainty
        self.wo_uncertainty = False
        
        self.netD = Discriminator(in_channels=3, use_sigmoid=(gan_type!='hinge'))

        # main loss
        self.reconstruction_loss = ReconstructionLoss(loss_weights['lambda_hole'], loss_weights['lambda_valid'])
        self.adversarial_loss = AdversarialLoss(type=gan_type)

        # auxiliary loss
        self.vgg_feature = VGG19() if (self.loss_weights['lambda_style'] or self.loss_weights['lambda_percep']) else None
        self.style_loss = StyleLoss() if self.loss_weights['lambda_style'] != 0 else None
        self.perceptual_loss = PerceptualLoss() if self.loss_weights['lambda_percep'] != 0 else None
        self.tv_loss = TotalVariationLoss(c_img) if self.loss_weights['lambda_tv'] != 0 else None

        self.writer = SummaryWriter(logPath)
    
    def forward(self, pred_imgs, comp_imgs, gt_imgs, masks, uncertainty, lambda_uncertainty, mutual_info, count, epoch, stable_training=True, current_step=1):
        # adversarial loss
        self.netD.zero_grad()
        gen_adv_loss, dis_loss = self.calculate_adv_loss(comp_imgs, gt_imgs)

        if stable_training: 
            uncertainty = uncertainty.exp() # from relu output [0, +inf] to [1, +inf]
        # reconstruction loss
        if self.wo_uncertainty:
            loss_rec = self.reconstruction_loss(pred_imgs, gt_imgs, masks, None)
            uncertainty_reg = torch.tensor(0.)
        else:
            uncertainty_log = torch.log(uncertainty)
            loss_rec = self.reconstruction_loss(pred_imgs, gt_imgs, masks, uncertainty)
            uncertainty_reg = torch.mean(uncertainty_log) * lambda_uncertainty
        l1_dist = torch.abs(pred_imgs.detach() - gt_imgs.detach()).mean()        
        
        # mutual information loss
        loss_mutual_info = mutual_info.mean() * self.loss_weights['lambda_mutal_info'] 

        # auxiliary loss(es)
        vgg_pred, vgg_gt = self.get_vgg_feature(pred_imgs, gt_imgs)
        loss_style = self.calculate_loss(self.style_loss, self.loss_weights['lambda_style'], vgg_pred, vgg_gt)
        loss_percep = self.calculate_loss(self.perceptual_loss, self.loss_weights['lambda_percep'], vgg_pred, vgg_gt)
        loss_tv = self.calculate_loss(self.tv_loss, self.loss_weights['lambda_tv'], pred_imgs, masks)

        gen_loss = loss_rec + gen_adv_loss + uncertainty_reg + loss_style + loss_percep + loss_tv + loss_mutual_info
        gen_loss = loss_rec + gen_adv_loss + uncertainty_reg + loss_style + loss_percep + loss_tv

        # summary
        self.writer.add_scalar('LossG/Adversarial loss_{}'.format(current_step), gen_adv_loss.item(), count)
        self.writer.add_scalar('LossD/Discrinimator loss_{}'.format(current_step), dis_loss.item(), count)
        self.writer.add_scalar('LossG/Reconstruction loss_{}'.format(current_step), loss_rec.item(), count)
        self.writer.add_scalar('LossG/L1 distance_{}'.format(current_step), l1_dist.item(), count)
        self.writer.add_scalar('LossG/style loss_{}'.format(current_step), loss_style.item(), count)
        self.writer.add_scalar('LossG/percep loss_{}'.format(current_step), loss_percep.item(), count)
        self.writer.add_scalar('LossG/tv loss_{}'.format(current_step), loss_tv.item(), count)
        self.writer.add_scalar('LossG/mutual info loss_{}'.format(current_step), loss_mutual_info.item(), count)
        self.writer.add_scalar('LossG/uncertainty regularization_{}'.format(current_step), uncertainty_reg.item(), count)

        if count % 200 == 0:
            print("loss_rec: ", loss_rec.item(), 
                  " uncertainty_reg: ", uncertainty_reg.item(), 
                  " gen_adv_loss: ", gen_adv_loss.item(),
                  " l1_dist: ", l1_dist.item(),
                  " loss_style: ", loss_style.item(), 
                  " loss_percep: ", loss_percep.item(), 
                  " loss_tv: ", loss_tv.item(),
                  " loss_mutual_info: ", loss_mutual_info.item(),
                  " dis_loss: ", dis_loss.item())
        
        return gen_loss, dis_loss
    
    def get_vgg_feature(self, pred_img, gt_img):
        if self.loss_weights['lambda_style'] !=0 or self.loss_weights['lambda_percep'] != 0:
            vgg_pred = self.vgg_feature(pred_img)
            vgg_gt = self.vgg_feature(gt_img)
        else:
            vgg_pred = None
            vgg_gt = None
        return vgg_pred, vgg_gt

    def calculate_loss(self, loss_func, loss_weights, *inputs):
        if loss_func is None:
            assert loss_weights == 0, "Error, you forget to pass by the loss function"
            return torch.tensor(0)
        return loss_func(*inputs) * loss_weights

    def calculate_adv_loss(self, comp_imgs, gt_imgs):

        if self.gan_type != 'wgan':
        # if self.gan_type == 'hinge':
            
            # discriminator loss
            dis_real_feat = self.netD(gt_imgs)                   
            dis_fake_feat = self.netD(comp_imgs.detach())         
            dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2
            
            # generator adversarial loss    
            gen_fake_feat = self.netD(comp_imgs)
            gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False)
            gen_adv_loss = gen_fake_loss * self.loss_weights['lambda_adv']
        else:
            raise NotImplementedError
        return gen_adv_loss, dis_loss

if __name__ == "__main__":
    a = 1
    b = 2
    from math import *
    def add(a, b):
        return a+b
    print(calculate_loss(sqrt, 1, 4))
    print(calculate_loss(add, 1, 4,6))
