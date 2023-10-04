import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, root_dir)
from networks.blocks import ConvBlock, DeconvBlock, MeanShift, norm, activation
from utils.utils import ResnetBlock

class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = self.conv_same_pad(kernel_size, stride)
        if type(padding) is not tuple:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding*2, 0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
            )

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding)

    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def forward(self, x):
        return self.trans_conv(x)

class UpBlock(nn.Module):

    def __init__(self, mode='bilinear', scale=2, channel=None, kernel_size=4):
        super().__init__()

        self.mode = mode
        if mode == 'deconv':
            self.up = ConvTranspose2dSame(
                channel, channel, kernel_size, stride=scale)
        else:
            def upsample(x):
                return F.interpolate(x, scale_factor=scale, mode=mode)
            self.up = upsample

    def forward(self, x):
        return self.up(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class BlendBlock(nn.Module):
    def __init__(
            self, c_in, c_out, ksize_mid=3, norm_type='in', act_type='relu'):
        super(BlendBlock, self).__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = nn.Sequential(
            Conv2dSame(c_in, c_mid, 1, 1),
            norm(c_mid, norm_type),
            activation(act_type),
            Conv2dSame(c_mid, c_out, ksize_mid, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.blend(x)

class FusionBlock(nn.Module):
    def __init__(self, c_feat_en, c_feat_de, c_feat_out):
        super(FusionBlock, self).__init__()
        # channel attention
        self.cal_en = nn.Sequential(
            CALayer(c_feat_en),
            nn.Conv2d(c_feat_en, c_feat_out, kernel_size=1),
        )

        self.cal_de = nn.Sequential(
            CALayer(c_feat_de),
            nn.Conv2d(c_feat_de, c_feat_out, kernel_size=1),
        )
        
        # spatial attention
        self.blend = BlendBlock(c_feat_out*2, 1)

    def forward(self, feat_en, feat_de):
        feat_en = self.cal_en(feat_en)
        feat_de = self.cal_de(feat_de)
        spatial_map = self.blend(torch.cat([feat_en, feat_de], dim=1))
        result = spatial_map * feat_en + (1 - spatial_map) * feat_de 
        return result, spatial_map

class DecodeBlock(nn.Module):
    def __init__(
            self, c_from_up, c_from_down, c_out, mode='bilinear',
            kernel_size=4, scale=2, norm_type='in', act_type='relu', fusion_mode='skip'):
        super(DecodeBlock, self).__init__()

        self.c_from_up = c_from_up     # from last decoder layer
        self.c_from_down = c_from_down # from encoder layer 
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpBlock(mode, scale, c_from_up, kernel_size=scale)

        if fusion_mode == "skip":
            self.fusion = Conv2dSame(self.c_in, self.c_out, kernel_size, stride=1)
        elif fusion_mode == 'att':
            self.fusion = FusionBlock(c_from_down, c_from_up, c_out)
        self.fusion_mode = fusion_mode

        layers = []
        layers.append(norm(self.c_out, norm_type=norm_type))
        layers.append(activation(act_type))
        self.post_ = nn.Sequential(*layers)
    
    def forward(self, x, concat=None):
        out = self.up(x)
        if self.fusion_mode == 'skip':
            out = self.fusion(torch.cat([out, concat], dim=1))
        elif self.fusion_mode == 'att':
            out, _ = self.fusion(concat, out)
        out = self.post_(out)
        return out

class SPADE(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(SPADE, self).__init__()
        self.param_free_norm = norm(in_channels, norm_type)
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            activation(act_type)
        )

        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.mlp_beta = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, modulation):
        normalized = self.param_free_norm(x)
        
        actv = self.mlp_shared(modulation)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # print(gamma.shape)
        # print(beta.shape)
        # print(normalized.shape)
        # exit()

        out = normalized * (1 + gamma) + beta
        return out


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_steps, residual_blocks=4, act_type='relu', norm_type='in', compress_mode='conv', fusion_mode='skip'):
        super(FeedbackBlock, self).__init__()

        # begin feedback
        self.en4 = nn.Sequential(
            nn.Conv2d(num_features*2, num_features*4, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            norm(num_features*4, norm_type),
            activation(act_type)
        )

        self.en5 = nn.Sequential(
            nn.Conv2d(num_features*4, num_features*8, kernel_size=4, stride=2, padding=1),     # 32 -> 16
            norm(num_features*8, norm_type),
            activation(act_type)
        )

        blocks = []
        if residual_blocks==4:
            dialtion_list = [2, 4, 4, 8]
        elif residual_blocks==8:
            dialtion_list = [2, 4, 4, 8, 8, 4, 4, 2]
        for i in range(residual_blocks):
            block = ResnetBlock(num_features*8, dialtion_list[i], norm_type)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        
        # 16 -> 32
        self.de1 = DecodeBlock(c_from_up=num_features*8, c_from_down=num_features*4, c_out=num_features*4, norm_type=norm_type, act_type=act_type, fusion_mode=fusion_mode)
        
        # 32 -> 64
        self.de2 = DecodeBlock(c_from_up=num_features*4, c_from_down=num_features*2 , c_out=num_features*2, norm_type=norm_type, act_type=act_type, fusion_mode=fusion_mode)
        # end feed back

        if compress_mode == 'conv':
            self.compress_in = nn.Sequential(
                nn.Conv2d(num_features*4, num_features*2, kernel_size=1),                
                norm(num_features*2, norm_type),
                activation(act_type)
            )
        elif compress_mode == 'spade':
            self.compress_in = SPADE(num_features*2, num_features*2, norm_type=norm_type, act_type=act_type)

        self.compress_mode = compress_mode

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).to(x.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        if self.compress_mode == 'conv':
            x = torch.cat((x, self.last_hidden), dim=1)# initial step concate 0 tensor
            x = self.compress_in(x)
        elif self.compress_mode == 'spade':
            x = self.compress_in(x, self.last_hidden)

        en4_out = self.en4(x)
        en5_out = self.en5(en4_out)
        middle_out = self.middle(en5_out)
        de1_out = self.de1(middle_out, concat=en4_out)
        de2_out = self.de2(de1_out, concat=x)

        self.last_hidden = de2_out

        return de2_out

    def reset_state(self):
        self.should_reset = True

class Inpaint_v1(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, residual_blocks=8, act_type='relu', norm_type='in', compress_mode='conv', fusion_mode='skip', use_unsup3d=False):
        super(Inpaint_v1, self).__init__()
        self.final_uncertainty_act = nn.Softplus() if use_unsup3d else nn.ReLU(True) 
        self.en1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, num_features//2, kernel_size=7, padding=0),
            norm(num_features, norm_type),
            activation(act_type)
        )

        self.en2 = nn.Sequential(
            nn.Conv2d(num_features//2, num_features, kernel_size=4, stride=2, padding=1),
            norm(num_features, norm_type),
            activation(act_type)
        )

        self.en3 = nn.Sequential(
            nn.Conv2d(num_features, num_features*2, kernel_size=4, stride=2, padding=1),
            norm(num_features*2, norm_type),
            activation(act_type)
        )

        # begin feedback 
        self.feed_back = FeedbackBlock(num_features, num_steps, residual_blocks=residual_blocks, act_type=act_type, norm_type=norm_type, compress_mode=compress_mode)
        # end feedback
        
        self.de3 = DecodeBlock(c_from_up=num_features*2, c_from_down=num_features, c_out=num_features, norm_type=norm_type, act_type=act_type, fusion_mode=fusion_mode)

        self.de4 = DecodeBlock(c_from_up=num_features, c_from_down=num_features//2, c_out=num_features//2, norm_type=norm_type, act_type=act_type, fusion_mode=fusion_mode)

        self.rgb_final = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(in_channels=num_features//2, out_channels=3, kernel_size=7, padding=0)
        )

        self.uncertainty_final = nn.Sequential(
            nn.Conv2d(in_channels=num_features//2, out_channels=1, kernel_size=1, padding=0),
            self.final_uncertainty_act
        )

        self.num_steps = num_steps

    def forward(self, x, mask):
    # def forward(self, x):
        self._reset_state()
        
        en1_out = self.en1(x)                                           #  256 -> 256
        en2_out = self.en2(en1_out)                                     #  256 -> 128
        en3_out = self.en3(en2_out)                                     #  128 -> 64
        rgb_outs = []       
        uncertainty_outs = []
        for _ in range(self.num_steps):
            de2_out = self.feed_back(en3_out)                           # 64 -> 64

            de3_out = self.de3(de2_out, concat=en2_out)                 # 64 -> 128
            de4_out = self.de4(de3_out, concat=en1_out)                 # 128 -> 256
            
            rgb_out = self.rgb_final(de4_out)
            uncertainty_out = self.uncertainty_final(de4_out)
            rgb_outs.append(torch.tanh(rgb_out))     # tanh: [-1, 1]
            uncertainty_outs.append(uncertainty_out) # relu
            
        return rgb_outs, uncertainty_outs, [torch.tensor(0.).to(uncertainty_out.device)]*len(uncertainty_outs) # return output of every timesteps

    def _reset_state(self):
        self.feed_back.reset_state()

if __name__ == '__main__':
    import numpy as np

    net = Inpaint_v1(in_channels=4,
                     out_channels=4,
                     num_features=96,
                     num_steps=5,
                     residual_blocks=8,
                     act_type="relu",
                     norm_type='in',
                     compress_mode='spade',
                     fusion_mode='att',
                     use_unsup3d=False,
                     ).cuda()
    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(model)
    print('Total trainable tensors: {}M'.format(num / 1e6))
    print(net)
    img = torch.randn(4, 4, 256, 256).cuda()
    mask = torch.randn(4, 3, 256, 256).cuda()
    y, _ = net(img, mask)
    # print(y.size())
    # feed = FeedbackBlock().cuda()
    # fe = torch.randn(4, 64, 128, 128).cuda()
    # hidden = torch.randn(4, 64, 128, 128).cuda()
    # out = feed(fe, hidden)
    # print(out.shape)