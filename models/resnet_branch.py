from copy import  deepcopy
from turtle import forward
import torch
from torch import Tensor, unsqueeze
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from .classifier import cosine_classifer

import pdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Mulvector(nn.Module):
    def __init__(self,vec_dim):
        super(Mulvector, self).__init__()
        self.lamda=nn.Parameter(torch.ones(vec_dim))

    def forward(self, x):
        #self.lamda = self.lamda/torch.norm(self.lamda,p=2)
        #self.lamda = torch.argmax(self.lamda,0)
        # x = x*self.lamda
        return x
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim,input_dim)
        self.fc2 = nn.Linear(input_dim,output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.in1 = torch.nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.in2 = torch.nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.in2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        classifier = 'linear'
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        # self.in1 = torch.nn.InstanceNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)
        self.fc_inv =  nn.Linear(512 * block.expansion, num_classes)
        self.fc_spe =  nn.Linear(512 * block.expansion, num_classes)
        # if classifier == 'linear':
        #     self.classifier = nn.Linear(512 * block.expansion, num_classes)
        # elif classifier == 'cosine':
        #     self.classifier = cosine_classifer(512 * block.expansion, num_classes)
        # elif classifier == 'mlp':
        #     self.classifier = MLP(512 * block.expansion, num_classes)
        # else:
        #     self.classifier = lambda x : x
        self.mul = Mulvector(512 * block.expansion)
        self.eps = 1e-5
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.in1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.unistyle(x,self.eps)
        x = self.layer2(x)
        # x = self.unistyle(x,self.eps)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_inv = self.mul(x)
        x_spe =x - x_inv
        logit_inv = self.fc_inv(x_inv)
        logit_spe = logit_inv - logit_inv

        return torch.stack([logit_inv, logit_spe])

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def unistyle(self,x,eps:Tensor) ->Tensor: 
        x_mean = x.mean(dim=-1).mean(dim=-1).detach()
        x_var = ((x**2).mean(dim=-1).mean(dim=-1) - x_mean**2    + eps).detach()
        x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
        x_var = x_var.unsqueeze(-1).unsqueeze(-1)
        x = (x-x_mean)/x_var.sqrt()
        return x
        
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.unistyle(x,self.eps)
        x = self.layer2(x)
        # x = self.unistyle(x,self.eps)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        feature_vec=x
        
        x = self.classifier(x)

        return x, feature_vec

class PartialSharedResNet(nn.Module):
    def __init__(self,model,branch_num,shared_keys):
        super().__init__()
        models = [deepcopy(model) for i in range(branch_num)]
        #parameter shared
        self.shared_names=[]
        self.branch_num =branch_num
    
        #共享参数依赖于命名 若命名里没有conv则不能共享，如downdsample里有Conv2d 不过可以通过'downsample.0'找到这个参数  共享层则可以做到 可以通过isinstance(layer,nn.Conv2d)判断要共享的卷积层
        #共享层也可以通过类似共享参数的关键词的方式来共享，其名字的例子：'layer4.1.relu', 'layer4.1.conv2'
        # self.shared_params=[]
        # for name,param in model.named_parameters():   #_parameters属性为只属于本模块的参数，如conv1._parameters() 是weight和bias，resnet则为[]
        #     for key in shared_keys:
        #         if key in name:
        #             self.shared_names.append(name)
        #             self.shared_params.append(param)  
        # print(f'{shared_keys}') 
        # print(f'{self.shared_names}') 

        self.shared_layers_names=[] 
        self.shared_layers=[]
        for name,layer in model.named_modules():
            for key in shared_keys:
            # if isinstance(layer,( nn.MaxPool2d,nn.AdaptiveAvgPool2d,nn.ReLU)) :
                if key in name:
                    self.shared_layers_names.append(name)
                    self.shared_layers.append(layer)   
        print(f'{shared_keys}') 
        print(f'{self.shared_layers_names}')                       
        for m in models:
            # self.change_param_to_shared(self.shared_names,self.shared_params,m)
            self.change_layer_to_shared(self.shared_layers_names,self.shared_layers,m)

        self.models = nn.Sequential(*models)

    def forward(self,samples):
        outs=[]
        for idx,x in enumerate(samples):
            out = self.models[idx](x)
            outs.append(out)
        return outs
    
    def change_layer_to_shared(self,shared_layers_names,shared_layers,model):
        named_modules= [name for name,layer in model.named_modules()]
        for name in named_modules:
            if name in shared_layers_names:
                idx = shared_layers_names.index(name)
                self.input_layer(model,name,shared_layers[idx])
    
    def input_layer(self,model,name:str,layer):
        def set_layer_by_name(module,name:str,layer):
            if name.isdigit():
                    module[int(name)] = layer
            else:
                module.__setattr__(name,layer)

                    
        def get_layer_by_name(module,name:str):
            if name.isdigit():
                return module[int(name)]
            else:
                return module.__getattr__(name)  
        names= name.split('.')
        if len(names)==1:
            set_layer_by_name(model,names[0],layer)
        elif len(names)==2:
            m = get_layer_by_name(model,names[0])
            set_layer_by_name(m,names[1],layer)
        elif len(names)>2:
            m = get_layer_by_name(model,names[0])
            for n in names[1:-1]:
                m = get_layer_by_name(m,n)  #__getattr__ and __setattr__ aquire value or set value for name only consticted to attr belong to this layer
            set_layer_by_name(m,names[-1],layer)
        else:
            NotImplementedError('names = []')
        
 


    def change_param_to_shared(self,shared_names,shared_params,model):                         #error: 'parameter name can\'t contain "."'
        for name,param in model.named_parameters():
            if name in shared_names:
                idx = shared_names.index(name)
                self.input_param_value(model,name,shared_params[idx])

    
    def input_param_value(self,model,name:str,param):
        names= name.split('.')
        if len(names)==1:
            model.__setattr__(names[0],param)#names[0] can't be digit
        elif len(names)==2:
            m = model.__getattr__(names[0]) #names[0] and names[1] can't be digit names[1]肯定不会  names[0]要注意
            m.__setattr__(names[1],param)
        elif len(names)>2:
            m = model.__getattr__(names[0])
            for n in names[1:-1]:
                if n.isdigit():          #so nembler of nn.sequentials can't >10
                    m = m[int(n)]
                else:
                    m = m.__getattr__(n)  #__getattr__ and __setattr__ aquire value or set value for name only consticted to attr belong to this layer
            m.__setattr__(names[-1],param) #names[-1] can't be digit  parameter肯定不会 因为一般是weight啥的
        else:
            NotImplementedError('names = []')



def shared_resnet18(brach_num=3,shared_keys=None,**kwargs):
    model = resnet18(**kwargs)
    shared_model = PartialSharedResNet(model,brach_num,shared_keys)
    return shared_model


def shared_resnet50(brach_num=3,shared_keys=None,**kwargs):
    model = resnet50(**kwargs)
    shared_model = PartialSharedResNet(model,brach_num,shared_keys)
    return shared_model
    
def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
