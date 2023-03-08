# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from copy import deepcopy
import os
from cv2 import transform
import torch
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from torch.utils.data.dataset import ConcatDataset
from data import datasets
from data.datasets import PACS, PACS_NOSPLIT,PACS_SPL_TR,PACS_SPL_VAL,CustomConcatDataset
from data.fast_data_loader import FastDataLoader
from .samplers import SubsetRandomSampler

import pdb

import data


try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


def build_loader(config,target_idx):

    hparams={'split':config.DATA.SPLIT,'split_rate':config.DATA.SPLIT_RATE,'aug':config.AUG.AUTO_AUGMENT}
    dataset = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,[target_idx], hparams)
    # [[train1,val1],[train2,val2],target,[train3,val3]]  and position of target is corresponding to target domian idx
    if config.DATA.SPLIT == '':
        hparams_val={'split':config.DATA.SPLIT,'split_rate':config.DATA.SPLIT_RATE,'aug':'noaug'}
        if config.DATA.DATASET != 'PACS':
            dataset = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,[target_idx], hparams)
            dataset_val = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,[target_idx], hparams_val)
        else:
            dataset = PACS_NOSPLIT(config.DATA.DATA_PATH,[target_idx], hparams)
            dataset_val = PACS_NOSPLIT(config.DATA.DATA_PATH,[target_idx], hparams_val)

        [dataset_src_trs,dataset_tar]=dataset.get_datasets()   
        [dataset_src_vals,dataset_tar]=dataset_val.get_datasets() 

    else:
        [[dataset_src_trs,dataset_src_vals],dataset_tar]=dataset.get_datasets()

    # dataset_src_vals= dataset_src_trs
    dataloaders_src_tr= datasetsToloaders(dataset_src_trs,config.DATA.BATCH_SIZE,drop_last=True,config=config)
    ###note shuffle！
    dataloaders_src_val = datasetsToloaders(dataset_src_vals,config.DATA.TEST_BATCH_SIZE,drop_last=False,config=config)
    dataloaders_tar = datasetsToloaders(dataset_tar,config.DATA.TEST_BATCH_SIZE,drop_last=False,config=config)[0]
    num_steps_train = get_numsteps_from_loaders(dataloaders_src_tr)
    num_steps_val = get_numsteps_from_loaders(dataloaders_src_val)
    return  dataset_tar, dataloaders_src_tr, dataloaders_src_val,dataloaders_tar, num_steps_train,num_steps_val

def build_distil_loader(config,target_idx):

    hparams={'split':config.DATA.SPLIT,'split_rate':config.DATA.SPLIT_RATE,'aug':config.AUG.AUTO_AUGMENT}
    # [[train1,val1],[train2,val2],target,[train3,val3]]  and position of target is corresponding to target domian idx
    if config.DATA.SPLIT == '':
        hparams_val={'split':config.DATA.SPLIT,'split_rate':config.DATA.SPLIT_RATE,'aug':'noaug'}
        if config.DATA.DATASET != 'PACS':
            dataset = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,[target_idx], hparams)
            dataset_val = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,[target_idx], hparams_val)
        else:
            dataset = PACS_NOSPLIT(config.DATA.DATA_PATH,[target_idx], hparams)
            dataset_val = PACS_NOSPLIT(config.DATA.DATA_PATH,[target_idx], hparams_val)

        [dataset_src_trs,dataset_tar] = dataset.get_datasets()
        [dataset_src_vals,dataset_tar]=dataset_val.get_datasets()

    else:
        dataset = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,[target_idx], hparams)
        [[dataset_src_trs,dataset_src_vals],dataset_tar] = dataset.get_datasets()

    # dataset_src_vals = dataset_src_trs
    dataset_train_sum = sum_datasets(dataset_src_trs,with_domain_label=True)
    dataset_val_sum = sum_datasets(dataset_src_vals,with_domain_label=True)
    if config.TRAIN.RANDOM_SAMPLER:
        data_loader_train = datasetsToloaders(dataset_train_sum,config.DATA.BATCH_SIZE,drop_last=True,config=config)
    else:
        data_loader_train= datasetsToloaders(dataset_src_trs,config.DATA.BATCH_SIZE,drop_last=True,config=config)
    data_loader_val = datasetsToloaders(dataset_val_sum,config.DATA.TEST_BATCH_SIZE,drop_last=False,config=config)
    data_loader_tar = datasetsToloaders(dataset_tar[0],config.DATA.TEST_BATCH_SIZE,drop_last=False,config=config)

    return dataset_train_sum, dataset_val_sum, data_loader_train, data_loader_val,data_loader_tar


def build_analy_loader(config,target_idx):
    
    hparams={'split':config.DATA.SPLIT,'split_rate':config.DATA.SPLIT_RATE,'aug':config.AUG.AUTO_AUGMENT}      #spetial   
    # [[train1,val1],[train2,val2],target,[train3,val3]]  and position of target is corresponding to target domian idx
    dataset = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,
        list(range(len(config.DATA.DOMAINS))), hparams)
    [[_,_],dataset_all] = dataset.get_datasets()
    tar_dataset = dataset_all[target_idx]
    dataset_all.pop(target_idx)
    sou_datasets=dataset_all
    sou_dataset_agg = sum_datasets(sou_datasets,with_domain_label=True)
    # dataset_sum = sum_datasets(dataset_all)
    tar_loader = datasetsToloaders(tar_dataset,config.DATA.TEST_BATCH_SIZE,drop_last=False,config=config)
    sou_loader = datasetsToloaders(sou_dataset_agg,config.DATA.BATCH_SIZE,drop_last=False,config=config)

    return sou_loader,tar_loader

def build_mixup_fn(config):
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    return mixup_fn


def datasetsToloaders(datasets,batchsize,drop_last,config):
    if  isinstance(datasets,list):
        data_loaders=[]
        for dataset in  datasets:
            data_loader = FastDataLoader(
                dataset,
                batch_size=batchsize,

                num_workers=config.DATA.NUM_WORKERS,

                drop_last=drop_last,
            )
            data_loaders.append(data_loader)
    else :
        data_loaders = FastDataLoader(
                datasets,
                batch_size=batchsize,

                num_workers=config.DATA.NUM_WORKERS,

                drop_last=drop_last,
            )
    
    return data_loaders

def sum_datasets(datasets,with_domain_label=False):
    if with_domain_label:
        datasetSum = CustomConcatDataset(datasets)
    else:
        for idx,dataset in enumerate(datasets):
            if idx == 0:
                datasetSum = dataset
            else:
                datasetSum = datasetSum + dataset
    return datasetSum
        


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_numsteps_from_loaders(dataloaders):
    numsteps = 1e10
    for dataloader in dataloaders:
        length = len(dataloader) 
        if length < numsteps:
            numsteps = length
    return numsteps
