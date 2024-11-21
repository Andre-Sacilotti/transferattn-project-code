import torch
import torch.nn as nn
import torch.nn.functional as F
from math import factorial
import numpy as np
from sklearn.manifold import TSNE
import math
import torch
from torch import nn
from torch.nn import functional as F

from torch.cuda.amp import custom_bwd, custom_fwd



class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def clamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

def sample_center_correntropy(x,y,kernel_size):
    """ Computing the center correntropy between two vectors x and y
    ----------
    x : np.array
        the first sample
    y : np.array
        the second sample
    kernel_size: float
        the kernel width
        
    Returns
    -------
      : float
        center correntropy between X and Y
    """    

    twosquaredSize = 2*kernel_size**2
    bias = 0
    for i in range(x.shape[0]):
        bias +=sum(torch.exp(-(x[i]-y)**2/twosquaredSize))
        #for j in range(x.shape[0]):
        #    bias +=np.exp(-(x[i]-y[j])**2/twosquaredSize)
    bias = bias/x.shape[0]**2
    
    corren = (1/x.shape[0]) * sum(torch.exp(-(x-y)**2/twosquaredSize)) -bias
    return corren


class _AutoDIALBatchNorm(nn.Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 
                     'num_domains', 'num_features', 'affine']

    def __init__(self,
                 num_domains,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 alpha_condition=True
        ):
        super(_AutoDIALBatchNorm, self).__init__()

        self.num_domains = num_domains
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.alpha_condition = True

        self.alpha = nn.Parameter(torch.Tensor(num_domains, num_features))
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_domains+1, num_features))
            self.register_buffer('running_var', torch.ones(num_domains+1, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        nn.init.ones_(self.alpha)
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        s = ('{num_domains}, {num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}')
        return s.format(**self.__dict__)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        # store the shape
        shape = input.size()

        # clip alpha in the range [0.0, 1.0]
        alpha = clamp(self.alpha, min=0.0, max=1.0)

        # collapse all the dimensions except N x D and C from (N x D, C, ...)
        x = input.view(shape[:2] + (-1, ))

        if bn_training:

            # calculate the global statistics
            var_g, mean_g = torch.var_mean(x, dim=[0, 2])

            # keep running statistics (moving average of mean and var)
            if self.track_running_stats:
                self.running_mean[self.num_domains] = exponential_average_factor * self.running_mean[self.num_domains] + (1 - exponential_average_factor) * mean_g.detach()
                self.running_var[self.num_domains] = exponential_average_factor * self.running_var[self.num_domains] + (1 - exponential_average_factor) * var_g.detach()

        else:
            mean_g = self.running_mean[self.num_domains]
            var_g = self.running_var[self.num_domains]

        # reshape (N x D, C, ...) to (N, D, C, ...)
        x = x.view((-1, self.num_domains) + x.shape[1:])

        # transpose (N, D, C, ...) to (D, N, C, ...)
        x = x.transpose(0, 1)
        norm = []
        corr_coef = 0
        corr_type = self.corr_type_autodial
        # print("aaaa", x[0].shape)
        with torch.no_grad():
            corr_coef_ = sample_center_correntropy(x[0].cpu().detach(), x[1].cpu().detach(), 10).to('cuda')
            # print('nanaaaaaaaaaaaaaaaa', x[0])
            # print("vall antes", corr_coef_)
            # corr_coef_ = torch.stack([sample_center_correntropy(i.cpu(), j.cpu(), 10) for i,j in zip(, x[1])])
            if corr_type == 'min':
                corr_coef += torch.min(torch.min(corr_coef_))#*100
            elif corr_type == 'max':
                corr_coef += torch.max(torch.max(corr_coef_))#*100
            elif corr_type == 'mean':
                corr_coef += torch.mean(torch.mean(corr_coef_))#*100
            elif corr_type == 'sum':
                corr_coef += torch.sum(torch.sum(corr_coef_))#*100
            corr_coef = clamp(torch.abs(corr_coef), min=0, max=1)
            # print(corr_coef)
            corr_coef.requires_grad = False
       


           

        # print("valor da corr", corr_coef)
        for i in range(self.num_domains):
            # take the i-th domain
            x_i = x[i]

            if bn_training:

                # calculate the domain statistics
                var_i, mean_i = torch.var_mean(x_i, dim=[0, 2])

                # keep running statistics (moving average of mean and var)
                if self.track_running_stats:
                    self.running_mean[i] = exponential_average_factor * self.running_mean[i] + (1 - exponential_average_factor) * mean_i.detach()
                    self.running_var[i] = exponential_average_factor * self.running_var[i] + (1 - exponential_average_factor) * var_i.detach()

            else:
                mean_i = self.running_mean[i]
                var_i = self.running_var[i]

            if self.alpha_condition == 'default':
                # calculate the cross-domain statistics
                mean = alpha[i] * mean_i + (1 - alpha[i]) * mean_g
                var = alpha[i] * var_i + (1 - alpha[i]) * var_g
            elif self.alpha_condition == 'correlation1':
                mean = alpha[i] * mean_i + (1 - alpha[i]) * mean_g*corr_coef
                var = alpha[i] * var_i + (1 - alpha[i]) * var_g*corr_coef
            elif self.alpha_condition == 'correlation2':
                mean = mean_i + (corr_coef) * mean_g
                var = var_i + (corr_coef) * var_g
            elif self.alpha_condition == 'correlation3':
                mean = (1 - corr_coef) * mean_g  +  mean_i
                var = (1 - corr_coef) * var_g +  var_i
            elif self.alpha_condition == 'correlation4':
                mean = alpha[i] * mean_i*corr_coef + (1 - alpha[i]) * mean_g
                var = alpha[i] * var_i*corr_coef + (1 - alpha[i]) * var_g
            elif self.alpha_condition == 'correlation5':
                mean = mean_g + (corr_coef) * mean_i
                var = var_g + (corr_coef) * var_i
            elif self.alpha_condition == 'correlation6':
                mean = mean_g + (1-corr_coef) * mean_i
                var = var_g + (1-corr_coef) * var_i
            # print("mean var", mean, var)
            # normalize the domain activations
            norm_i = (x_i - mean[:, None]) / torch.sqrt(var[:, None] + self.eps)

            # store the normalized batch of the domain i in a list
            norm.append(norm_i)

        # stack the normalized batches of all the domains
        x = torch.stack(norm, dim=1)

        # reshape (N, D, C, ...) to (N x D, C, ...)
        x = x.view((-1, ) + x.size()[2:])

        # scale and shift the normalized activations
        if self.affine:
            x = x * self.weight[:, None] + self.bias[:, None]

        return x.view(shape)


class AutoDIALBatchNorm1d(_AutoDIALBatchNorm):

    def set_condition_type(self, type_, corr_type_autodial):
        self.alpha_condition = type_
        self.corr_type_autodial = corr_type_autodial

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class AutoDIALBatchNorm2d(_AutoDIALBatchNorm):


    def set_condition_type(self, type_, corr_type_autodial):
        self.alpha_condition = type_
        self.corr_type_autodial = corr_type_autodial

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class AutoDIALBatchNorm3d(_AutoDIALBatchNorm):


    def set_condition_type(self, type_, corr_type_autodial):
        self.alpha_condition = type_
        self.corr_type_autodial = corr_type_autodial

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))


class AutoDIAL(nn.Module):
    __constants__ = ['num_domains']

    def __init__(self, backbone, num_domains=2, **kwargs):
        super(AutoDIAL, self).__init__()

        if not isinstance(backbone, nn.Module):
             raise ValueError('A model must be provided.')

        self.backbone = backbone
        self.num_domains = num_domains

        autodial = dict()
        for name, layer in self.backbone.named_modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                if isinstance(layer, nn.BatchNorm1d):
                    autodial_bn = AutoDIALBatchNorm1d(self.num_domains, layer.num_features, **kwargs)
                elif isinstance(layer, nn.BatchNorm2d):
                    autodial_bn = AutoDIALBatchNorm2d(self.num_domains, layer.num_features, **kwargs)
                elif isinstance(layer, nn.BatchNorm3d):
                    autodial_bn = AutoDIALBatchNorm3d(self.num_domains, layer.num_features, **kwargs)
            
                state_dict = layer.state_dict()
               
                if autodial_bn.affine:

                    if 'weight' in state_dict.keys():
                        autodial_bn.weight.data.copy_(state_dict['weight'].data)

                    if 'bias' in state_dict.keys():
                        autodial_bn.bias.data.copy_(state_dict['bias'].data)

                if autodial_bn.track_running_stats:

                    if 'running_mean' in state_dict.keys():
                        for i in range(self.num_domains+1):
                            autodial_bn.running_mean[i].data.copy_(state_dict['running_mean'].data)
 
                    if 'running_var' in state_dict.keys():
                        for i in range(self.num_domains+1):
                            autodial_bn.running_var[i].data.copy_(state_dict['running_var'].data)

                    if 'num_batches_tracked' in state_dict.keys():
                        autodial_bn.num_batches_tracked.data.copy_(state_dict['num_batches_tracked'].data)

                autodial[name] = autodial_bn

        for key, value in autodial.items():
            rsetattr(self.backbone, key, value)

    def extra_repr(self):
        s = ('num_domains={num_domains}')
        return s.format(**self.__dict__)

    def forward(self, x):
        x = self.backbone(x)
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        self.apply(init_weights)

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

#特征提取
class FeatureEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        linear = nn.Linear(in_dim, out_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=True),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=True),
            nn.ReLU(True),
        )
        self.apply(init_weights)

    def forward(self, feats: torch.Tensor):
        N, T, C = feats.shape
        #feats = feats.contiguous()
        #print(C)
        feats = feats.view(-1, C)
        feats = self.encoder(feats)
        feats = feats.view(N, T, self.out_dim)

        return feats


class Classifier(nn.Module):
    def __init__(self, in_dim, cls_num):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifer = nn.Linear(in_dim, cls_num)
        self.apply(init_weights)

    def forward(self, feats: torch.Tensor):
        feats = feats.transpose(1, 2).contiguous()
        B = feats.shape[0]
        feats = self.pool(feats).view(B, -1)
        logit = self.classifer(feats)
        return feats, logit

class DiscrminiatorBlock(nn.Module):
    def __init__(self, in_dim, seg_sizes):
        super().__init__()
        self.seg_sizes = seg_sizes
        self.discriminator_groups = nn.ModuleList(
            nn.ModuleList([Discriminator(in_dim) for _ in range(s)])
            for _, s in enumerate(seg_sizes))

        self.global_discriminators = nn.ModuleList(
            [NDiscriminator(in_dim) for _ in range(len(seg_sizes))])

    def forward(self, inputs):
        assert len(inputs) == len(self.discriminator_groups)

        logit_group = []
        for g_id, group in enumerate(self.discriminator_groups):
            inp = inputs[g_id]
            logits = []
            for d_id, dis in enumerate(group):
                feats = inp[:, :, d_id]
                logit = dis(feats)
                logits.append(logit)
            logits = torch.cat(logits, dim=1)
            logit_group.append(logits)


        scale_w = []
        for g_id, dis in enumerate(self.global_discriminators):
            inp = inputs[g_id]
            logit = dis(inp)
            pred = torch.sigmoid(logit)
            dis = torch.cat([pred, 1 - pred], dim=-1)
            #print(dis)
            ent = - 1.0 * torch.sum(torch.log(dis) * dis, dim=-1, keepdim=True)
            scale_w.append(ent.detach())

        scale_w = torch.cat(scale_w, dim=1)
        sum_scale = (torch.sum(scale_w, dim=1).view(len(scale_w), 1)).expand([-1, len(self.seg_sizes)])
        scale_w = scale_w / sum_scale
        return logit_group, scale_w


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        DIM = 1024
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, 1),
        )
        self.apply(init_weights)

    def forward(self, feat):
        logit = self.classifer(feat)
        return logit


class GradReverse2(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

class NDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        DIM = 128
        self.classifer = nn.Sequential(
            # nn.Linear(in_dim, DIM),
            # nn.ReLU(True),
            # nn.Linear(DIM, DIM),
            # nn.ReLU(True),
            nn.Linear(in_dim, 1),
        )
        self.apply(init_weights)


    def forward(self, feat):
        B, C, _ = feat.shape
        feat = F.adaptive_avg_pool1d(feat, 1)
        feat = feat.view(B, C)
        logit = self.classifer(feat)
        return logit


class ResidualDialConvBlock(nn.Module):
    def __init__(self, in_dim, dilations, coeff_fn=lambda: 1, use_autodial=False, type_='alpha', corr_type_autodial='min'):
        super().__init__()
        self.net = nn.Sequential(
            *[ResidualDialConv(in_dim, i, use_autodial=use_autodial, type_=type_,corr_type_autodial=corr_type_autodial) for i in dilations])
        
    def forward(self, inp, domain=None, w=None):
        #inp = inp.transpose(1, 2).contiguous()
        itermidiate = []
        for layer in self.net:
            inp = layer(inp, domain, w)
            itermidiate.append(inp)
        return itermidiate


class ResidualDialConv(nn.Module):
    def __init__(self, in_dim, dilation, avg_pool=True, use_autodial=False, type_='alpha', corr_type_autodial='min'):
        super().__init__()
        k_size = 3
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=k_size, dilation=dilation),
            nn.ReLU(inplace=True), nn.Conv1d(in_dim, in_dim, kernel_size=1))
        self.un = nn.Unfold(kernel_size=[k_size, 1], dilation=dilation)
        self.k_size = k_size
        self.dilation = dilation
        self.apply(init_weights)
        if avg_pool:
            w = [1 / k_size ] * k_size
        else:
            w = list(range(k_size))
            mid = max(w) / 2
            w = [-1 * abs(i - mid) for i in w]
            w = np.exp(w) / sum(np.exp(w))
        w = torch.tensor(w).float()
        self.register_buffer('w', w)
        self.autodial_ = AutoDIALBatchNorm1d(2, 256)
        self.autodial_.set_condition_type(
            type_=type_,
            corr_type_autodial=corr_type_autodial
        )
        self.use_autodial = use_autodial

    def forward(self, inp, domain=None, w=None):
        B, C, _ = inp.shape
        #print(inp.shape)
        T = self.k_size
        conv = self.conv(inp)  # shape [B, C, S]
        seg = self.un(inp.unsqueeze(-1)).view(B, C, T, -1)  # shape [B, C, T, S]
        #w = w.transpose(0, 2).contiguous()
        #print(seg.shape)
        if w == None:
            w = self.w.view(1, 1, -1, 1).expand_as(seg)
        else:
            temp = torch.zeros([seg.shape[0], seg.shape[2], seg.shape[3]]).cuda()
            #print(self.dilation)
            sig = nn.Sigmoid()
            w = sig(w[0])
            #print(w)
            #w = 1 / (w * (1 - w))
            if domain == 'T':
                w = 1 / (1 - w)
            else:
                w = 1 / w
            for i in range(seg.shape[0]):
                for k in range(seg.shape[3]):
                    now = w[i][k] + w[i][k + self.dilation] + w[i][k + self.dilation * 2]
                    for j in range(seg.shape[2]):
                        # print(w[i][k + j * self.dilation] / now)
                        temp[i][j][k] = w[i][k + j * self.dilation] / now
            w = temp.view(seg.shape[0], -1, seg.shape[2], seg.shape[3]).expand_as(seg)
        #print(w)
        residual = torch.sum(seg * w, dim=2)
        out = conv + residual
        #out = out[ :, :, ::3]
        #print("laji", out.shape)
        # print("SAIDA AQUIIIIIIIIII", out.shape)
        # print("ANTES AUTODIAL out.shape)
        if self.use_autodial:
            out = self.autodial_(out)
            return out
        
        return out


# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb


class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )
            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck,self.num_class),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

class GradReverse2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class ClassifierRel2(nn.Module):
    def __init__(self, in_dim, cls_num):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifer = nn.Linear(in_dim, cls_num)
        self.apply(init_weights)

    def get_Uy(self):
        inters = self.classifer.weight
        Uy = self.classifer.weight.shape[0]*torch.inverse((inters.T@inters))
        return Uy

    def get_Wy(self):
        inters = self.classifer.weight.T
        return inters

    def forward(self, feats: torch.Tensor):
        feats = feats.transpose(1, 2).contiguous()
        B = feats.shape[0]
        feats = self.pool(feats).view(B, -1)
        logit = self.classifer(feats)
        return feats, logit


class ClassifierRel(nn.Module):
    def __init__(self, in_dim, cls_num):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifer = nn.Linear(in_dim, cls_num)
        self.apply(init_weights)

    def get_Uy(self):
        inters = self.classifer.weight
        Uy = self.classifer.weight.shape[0]*torch.inverse((inters.T@inters))
        return Uy

    def get_Wy(self):
        inters = self.classifer.weight.T
        return inters

    def forward(self, feats: torch.Tensor):
        logit = self.classifer(feats)
        return logit

class DomainClassifierRel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, K):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)


        self.classes = {i: nn.Linear(hidden_dim, 1).to('cuda') for i in range(K)}

        std = 0.001
        nn.init.normal_(self.fc1.weight, 0, std)
        nn.init.constant_(self.fc1.bias, 0)
        for k, v in self.classes.items():
            nn.init.normal_(v.weight, 0, std)
            nn.init.constant_(v.bias, 0)

    def get_Ud(self):
        inters = []
        for k, v in self.classes.items():
            inters.append(v.weight)
        inters = torch.cat(inters)
        Ud = self.classes[0].weight.shape[0]*torch.inverse((inters.T@inters))
        return Ud
    
    def get_Wd(self):
        inters = []
        for k, v in self.classes.items():
            inters.append(v.weight)
        inters = torch.cat(inters).T
        return inters


    def forward(self, x: torch.Tensor, k, coeff: float = 1.0):
        x = GradReverse2.apply(x, coeff)
        x = self.fc1(x)

        x = self.classes[k](x)


        return x



class ResClassifier(nn.Module):
    def __init__(self, class_num=12, extract=True, dropout_p=0.5, inputdim=512):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(inputdim, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        self.fc2 = nn.Linear(1000, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))            
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit         
        return logit


# def get_L2norm_loss_self_driven(x):
#     radius = x.norm(p=2, dim=1).detach()
#     assert radius.requires_grad == False
#     radius = radius + 1.0
#     l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
#     return 0.05 * l

def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2, dim=1).mean() - 25) ** 2
    return 0.05 * l
    
import torch
def calculate_LR(domain, classes):
    Wd = domain.get_Wd()
    Wy = classes.get_Wy()
    dd = Wd.shape[0]
    dy = Wy.shape[0]
    LR = torch.trace(Wd @ torch.inverse(Wy.T @ Wy) @ Wd.T) - (dd/dy) * (
        torch.log(torch.det(torch.matmul(Wd.T,Wd))) - torch.log(torch.det(torch.matmul(Wy.T,Wy)))
    )
    return LR

class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

        std = 0.001
        nn.init.normal_(self.fc1.weight, 0, std)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, std)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor, coeff: float = 1.0):
        x = GradReverse2.apply(x, coeff)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    m = ResidualDialConvBlock(256, [1, 3, 7, 15])
    print(m)