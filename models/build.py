# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from copy import deepcopy
from email.policy import strict
import torch
from torch import nn
from yaml import load_all
from torchvision import models
from .resnet_shared import ResNet,resnet18_shared,resnet50_shared
from .resnet import resnet18,resnet50
from .ensembles import Ensemble
from .ensembles import Distilation
from functools import partial
from .resnet_branch import shared_resnet18,shared_resnet50

import pdb



def build_model(config, load_ens=False):
    domain_num = len(config.DATA.DOMAINS)
    if load_ens:
        model_type = config.MODEL.ENSEM_TYPE
    else:
        model_type = config.MODEL.TYPE
    # if model_type == 'ensemble':
    #     model=Ensemble(model_fn, num_heads=config.ENS.NUM_HEADS,
    #                      num_classes=config.MODEL.NUM_CLASSES,
    #                      classifier =config.MODEL.CLASSIFIER,
    #                      prob=config.MODEL.ENS.HEAD_PROB,
    #                      dropout=config.MODEL.ENS.HEAD_DROPOUT)     
    if model_type == 'ResNet18_sh':
        model = resnet18_shared(pretrained=config.TRAIN.PRETRAINED,
                        num_classes=config.MODEL.NUM_CLASSES,
                        domain_num=len(config.DATA.DOMAINS)-1,
                        classifier=config.MODEL.CLASSIFIER)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,config.MODEL.NUM_CLASSES)
        model.copy_unshared_para()
    elif model_type == 'ResNet18_Branches':
        model = shared_resnet18(brach_num=domain_num-1,
                                shared_keys=config.MODEL.SHARED_KEYS,
                                pretrained=config.TRAIN.PRETRAINED,
                                num_classes=config.MODEL.NUM_CLASSES,
                                classifier=config.MODEL.CLASSIFIER)
    elif model_type == 'ResNet18':
        model = resnet18(pretrained=config.TRAIN.PRETRAINED,
                         classifier=config.MODEL.CLASSIFIER,
                         num_classes = config.MODEL.NUM_CLASSES)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,config.MODEL.NUM_CLASSES)
    elif model_type == 'cnn_digits':
        model = cnn_digitsdg()
    elif model_type == 'cnn_digits_sh':
        model = cnn_digitsdg_sh()
    elif model_type == 'ResNet50_sh':
        model = resnet50_shared(pretrained=config.TRAIN.PRETRAINED,num_classes=config.MODEL.NUM_CLASSES,
                        domain_num=len(config.DATA.DOMAINS)-1)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,config.MODEL.NUM_CLASSES)
        model.copy_unshared_para()
    elif model_type == 'ResNet50':
        model = resnet50(pretrained=config.TRAIN.PRETRAINED,
                         classifier=config.MODEL.CLASSIFIER,
                         num_classes = config.MODEL.NUM_CLASSES)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,config.MODEL.NUM_CLASSES)
    elif model_type == 'ResNet50_Branches':
        model = shared_resnet50(brach_num=domain_num-1,
                                shared_keys=config.MODEL.SHARED_KEYS,
                                pretrained=config.TRAIN.PRETRAINED,
                                num_classes=config.MODEL.NUM_CLASSES,
                                classifier=config.MODEL.CLASSIFIER)
    elif model_type == 'keyshare':
        model = resnet18(pretrained=config.TRAIN.PRETRAINED,
                    classifier=config.MODEL.CLASSIFIER,
                    num_classes = config.MODEL.NUM_CLASSES)
        model = build_shared_model(model,3,'conv')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model



def build_shared_model(model,branch_num,keys):
    models = [deepcopy(model) for _ in range(branch_num-1)]
    for m in models:
            for name,param in model.named_parameters():
                for key in keys:
                    if key in name:
                        m.named_parameters()[name] = param

