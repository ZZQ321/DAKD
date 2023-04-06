# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from copy import deepcopy
from modulefinder import Module
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.nn import CrossEntropyLoss
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from models.resnet_shared import ResNet
from torchvision import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from config import get_config
from models import build_model
from data import build_analy_loader
from lr_scheduler import build_scheduler
from models.ensembles import Distilation, Ensemble
from optimizer import build_optimizer
from logger import create_logger
from config import get_config,parse_option
from utils import tsne_plot_togeter,tsne_plot_separate
from utils import plot_embedding

# from domainbed.networks import ResNet as db_ResNet
from domainbed.algorithms import MMD
from domainbed import hparams_registry
from domainbed import algorithms

#from domain_train import classic_training,classic_test,classic_setting



import pdb



def main(config,args):
    writer = SummaryWriter(config.DISTILL_PATH)#,config.DATA.DOMAINS[target_idx]))
    domain_num = len(config.DATA.DOMAINS)   

    for target_idx in range(domain_num):
        sou_loader,tar_loader = build_analy_loader(config,target_idx) 
        os.makedirs(os.path.join(config.DISTILL_PATH,config.DATA.DOMAINS[target_idx],'analysis'), exist_ok=True)
        logger = create_logger(output_dir=os.path.join(config.DISTILL_PATH,config.DATA.DOMAINS[target_idx],'analysis') , name=f"{config.DATA.DOMAINS[target_idx]}")
        path = os.path.join(config.DISTILL_PATH,config.DATA.DOMAINS[target_idx],'analysis', "config.json")
        #write config
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(config.dump())
        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        
        dis_model = build_model(config)
        dis_model.cuda()
        dis_model_path = os.path.join(config.DISTILL_PATH,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
        dis_model.load_state_dict(torch.load(dis_model_path))
        logger.info(str(dis_model))
        ensem_model = build_model(config,load_ens=True) 
        ensem_model.cuda()
        ens_model_path = os.path.join(config.ENSEM_PATH,config.DATA.DOMAINS[target_idx], f"{config.DATA.DOMAINS[target_idx]}_model.pth")
        ensem_model.load_state_dict(torch.load(ens_model_path))

        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        db_haparams =  hparams_registry.default_hparams(args.algorithm, 'PACS')
        ctrmodel_path = os.path.join( 'domainbed/output/{}/{}'.format(args.algorithm,target_idx), 'model.pkl')
        ctrmodel_dict = torch.load(ctrmodel_path)['model_dict']
        ctr_algorithms = algorithm_class(input_shape=(3,224,224), num_classes=7, num_domains=3, hparams=db_haparams)
        ctr_algorithms.load_state_dict(ctrmodel_dict)
        ctr_algorithms.cuda()
        analysis_plot(sou_loader,tar_loader,ctr_algorithms,target_idx,model_name=(args.algorithm+''),model_type= 'distill')
        analysis_plot(sou_loader,tar_loader,dis_model,target_idx,model_name='dakd',model_type='distill')
        # analysis_plot(sou_loader,tar_loader,ensem_model,target_idx,'ens')
        
        
        # criterion = SoftTargetCrossEntropy()#torch.nn.CrossEntropyLoss()
        criterion = CrossEntropyLoss()#torch.nn.CrossEntropyLoss()
        criterion_distil = SoftTargetCrossEntropy()
        distill = Distilation(ensem_model,criterion_distil,config.TRAIN.T)

@torch.no_grad()
def analysis_plot(sou_loader,tar_loader,model,target_idx,model_name,model_type='distill'): 
    model.eval()   
    for idx,(samples,target) in enumerate(tar_loader):
        samples = samples.cuda(non_blocking=True)
        if model_type == 'distill':
            _,out = model.extract_features(samples)
            if idx==0:
                tar_X = out.cpu()
                tar_cata_label = target
            else:
                # break
                tar_X = torch.cat((tar_X,out.cpu()),0)
                tar_cata_label = torch.cat((tar_cata_label,target),0)
        elif model_type == 'ens':
            sou_num = len(config.DATA.DOMAINS) - 1
            samples = samples.repeat(sou_num,1,1,1,1)
            outs = model.extract_features(samples)
            sou_num = len(outs)
            B = outs[0].shape[0]
            out = torch.cat(outs)
            tar_exp_1 = torch.arange(0,sou_num).repeat(B,1).permute(1,0).flatten()
            target = target.repeat(sou_num)
            if idx==0:
                tar_X = out.cpu()
                tar_cata_label = target
                tar_expert = tar_exp_1
            else:
                # break
                tar_X = torch.cat((tar_X,out.cpu()),0)
                tar_cata_label = torch.cat((tar_cata_label,target),0)
                tar_expert = torch.cat((tar_expert,tar_exp_1),0)


    for idx,(samples,cata_label,db_label) in enumerate(sou_loader):
        samples = samples.cuda(non_blocking=True)
        if model_type == 'distill':
            _,out = model.extract_features(samples)
            if idx==0:
                sou_X = out.cpu()
                sou_db_label = db_label
                sou_cata = cata_label.cpu()
            else:
                # break
                sou_X = torch.cat((sou_X,out.cpu()),0)
                sou_db_label = torch.cat((sou_db_label,db_label),0)
                sou_cata = torch.cat((sou_cata,cata_label))

        elif model_type == 'ens':
            sou_num = len(config.DATA.DOMAINS) - 1
            samples = samples.repeat(sou_num,1,1,1,1)
            outs = model.extract_features(samples)
            sou_num = len(outs)
            B = outs[0].shape[0]
            out = torch.cat(outs)
            sou_exp_1 = torch.arange(0,sou_num).repeat(B,1).permute(1,0).flatten()
            db_label = db_label.repeat(sou_num)
            cata_label = cata_label.repeat(sou_num)
            if idx==0:
                sou_X = out.cpu()
                sou_db_label = db_label
                sou_cata = cata_label.cpu()
                sou_expert = sou_exp_1
            else:
                # break
                sou_X = torch.cat((sou_X,out.cpu()),0)
                sou_db_label = torch.cat((sou_db_label,db_label),0)
                sou_expert = torch.cat((sou_expert,sou_exp_1),0)
                sou_cata = torch.cat((sou_cata,cata_label))


    if model_type == 'distill':
        tsne_plot_togeter('bold_'+model_name+'/together',sou_X,sou_cata,sou_db_label,None,tar_X,tar_cata_label,None,target_idx)
        tsne_plot_separate('bold_'+model_name+'/separate',sou_X,sou_cata,sou_db_label,None,tar_X,tar_cata_label,None,target_idx)
    elif model_type == 'ens':
        tsne_plot_togeter('no_jitter',sou_X,sou_cata,sou_db_label,sou_expert,tar_X,tar_cata_label,tar_expert,target_idx)


    
    

        

if __name__ == '__main__':
    args, config = parse_option()
    seed = config.SEED 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(config,args)


#python analysis.py --cfg configs/tsne.yaml