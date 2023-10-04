import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil

from networks import create_model
from .base_solver import BaseSolver
from networks import init_weights
from utils import util
from solvers.InpaintingLoss import InpaintLoss, img2show

device = "cuda" if torch.cuda.is_available() else "cpu"

class InpaintSolver(BaseSolver):
    def __init__(self, opt):
        super(InpaintSolver, self).__init__(opt)

        # saving path configuration
        root_folder = opt['solver']['root_folder']
        exp_name = opt['solver']['exp_name']
        self.model_folder = os.path.join(root_folder, exp_name, 'models')
        self.results_folder = os.path.join(root_folder, exp_name, 'results')
        self.log_folder = os.path.join(root_folder, exp_name, 'logs')
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)
        
        # hyper parameter loading 
        self.solver_opt = opt['solver']
        self.num_steps = opt['networks']['num_steps']
        self.cl_weights = self.solver_opt['cl_weights']
        self.uncer_weights = self.solver_opt['uncer_weights']

        # build model
        # self.model = create_model(opt).cuda()
        self.model = create_model(opt).to(device)
        
        # build loss 
        self.inpaint_loss = InpaintLoss(logPath=self.log_folder, gan_type=self.solver_opt['gan_type'], loss_weights=self.opt["loss_weights"], wo_uncertainty=self.opt["networks"]['wo_uncertainty']).to(device)

        # build optimizer
        self.optim_D = torch.optim.Adam(self.inpaint_loss.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.solver_opt['learning_rate'], betas=(0.5, 0.9))

        # load model
        # self.load()

        # parallel
        self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.inpaint_loss = nn.DataParallel(self.inpaint_loss)
            
        self.print_network()

        self.count = 0
        self.current_epoch = 0

    def _net_init(self, init_type='normal'):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(self.model, init_type)

    def feed_data(self, batch, need_HR=True):
        self.masked_imgs, self.gt_imgs, self.masks = [_.to(device) for _ in batch]
        self.masked_imgs = self.masked_imgs * 2 - 1 # [0, 1] to [-1, 1]
        self.gt_imgs = self.gt_imgs * 2 - 1

        self.input_tensor = torch.cat((self.masked_imgs, self.masks), 1)

    # def test(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         pred_imgs, uncertainty_maps, mutual_infos = self.model(self.input_tensor, self.masks)
            
        
    def train_step(self):
        self.count += 1
        self.adjust_learning_rate()

        self.model.train()
        self.model.zero_grad()
        
        self.input_tensor = torch.cat((self.masked_imgs, self.masks), 1)
        pred_imgs, uncertainty_maps, mutual_infos = self.model(self.input_tensor, self.masks)
        comp_imgs = [pred_imgs[i] * self.masks + self.gt_imgs * (1 - self.masks) for i in range(len(pred_imgs))]

        gen_batch_loss = 0
        dis_batch_loss = 0
        for i in range(self.num_steps):
            gen_loss, dis_loss = self.inpaint_loss(pred_imgs[i], comp_imgs[i], self.gt_imgs, self.masks, uncertainty_maps[i], self.uncer_weights[i], mutual_infos[i], self.count, self.current_epoch, current_step=i+1)
    
            gen_batch_loss += self.cl_weights[i] * gen_loss.mean()
            dis_batch_loss += self.cl_weights[i] * dis_loss.mean()

        # update generator
        self.optim_G.zero_grad()
        gen_batch_loss.backward()
        self.optim_G.step()
    
        # update discriminator
        self.optim_D.zero_grad()
        dis_batch_loss.backward()
        self.optim_D.step()
    
        # add visualization
        if self.count % 100 == 0:
            for i in range(self.num_steps):
                img = img2show(self.gt_imgs, pred_imgs[i], comp_imgs[i], self.gt_imgs*(1-self.masks), normalize=True)
                uncertainty_vis = thutil.make_grid(uncertainty_maps[i][:4], normalize=True, scale_each=True)
                self.inpaint_loss.module.writer.add_image('train/whole_imgs_step_{}'.format(i+1), img, global_step=self.count)
                self.inpaint_loss.module.writer.add_image('train/uncertainty_step_{}'.format(i+1), uncertainty_vis, global_step=self.count)
            
        return gen_batch_loss

    def load(self):
        """
        load or initialize network
        """
        if self.solver_opt['pretrained_step'] != 0:
            gen_path = os.path.join(self.model_folder, 'step_'+str(self.solver_opt['pretrained_step'])+'_gen.pth')
            dis_path = os.path.join(self.model_folder, 'step_'+str(self.solver_opt['pretrained_step'])+'_dis.pth')
            opt_path = os.path.join(self.model_folder, 'step_'+str(self.solver_opt['pretrained_step'])+'_opt.pth')
            self.model.load_state_dict(torch.load(gen_path))
            print("Have loaded the Generator")
            try:
                self.inpaint_loss.netD.load_state_dict(torch.load(dis_path))
                opt_data = torch.load(opt_path)
                self.optim_G.load_state_dict(opt_data['optimG'])
                self.optim_D.load_state_dict(opt_data['optimD'])
                self.current_epoch = opt_data['epoch']
                self.count = opt_data['iteration']
                print("Successfully loaded model from {}, and current epoch is: {}, iteration is {}".format(gen_path, self.current_epoch, self.count))
            except:
                print("Warning: discriminator and optimizer are not loaded!")
        else:
            self._net_init()

    def save(self, enforce=False):
        if enforce or self.count % 2000 == 0:
            gen_path = os.path.join(self.model_folder, 'step_'+str(self.count)+'_gen.pth')
            dis_path = os.path.join(self.model_folder, 'step_'+str(self.count)+'_dis.pth')
            opt_path = os.path.join(self.model_folder, 'step_'+str(self.count)+'_opt.pth')
            
            torch.save(self.model.module.state_dict(), gen_path)
            torch.save(self.inpaint_loss.module.netD.state_dict(), dis_path)
            torch.save({'epoch': self.current_epoch, 'iteration': self.count, 'optimG': self.optim_G.state_dict(), 'optimD': self.optim_D.state_dict()}, opt_path)

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        print("==================================================")
    
    
    # get current learning rate
    def get_lr(self, type='G'):
        if type == 'G':
            return self.optim_G.param_groups[0]['lr']
        return self.optim_D.param_groups[0]['lr']
  
    # learning rate scheduler, step
    def adjust_learning_rate(self):
        print(self.solver_opt['niter_steady'])
        print(self.solver_opt['niter'])
        decay = 0.1 ** (min(self.count, self.solver_opt['niter_steady']) // self.solver_opt['niter']) 
        new_lr = self.solver_opt['learning_rate'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optim_G.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optim_D.param_groups:
                param_group['lr'] = new_lr
