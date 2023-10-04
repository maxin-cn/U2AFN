import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import use_spectral_norm

class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()
  
  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)



class Discriminator(BaseNetwork):
  """from PEN-Net, CVPR 2020"""
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      
      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)
    if init_weights:
      self.init_weights()


  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x

class old_Discriminator(nn.Module):
  """I may add some layers on this network"""
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(old_Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum,
        kernel_size=5, stride=1, padding=0, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      
      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*2,
        kernel_size=5, stride=1, padding=0, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*4,
        kernel_size=5, stride=1, padding=0, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)

  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x
