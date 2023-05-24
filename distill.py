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
from data.build import build_mixup_fn
from models.resnet_shared import ResNet
from torchvision import models

from config import get_config,parse_option
from models import build_model
from data import build_distil_loader
from lr_scheduler import build_scheduler
from models.ensembles import Distilation, Ensemble
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
#from domain_train import classic_training,classic_test,classic_setting
from tqdm import tqdm
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from torch.cuda.amp import GradScaler, autocast
DEBUG=0
import pdb



def main(config):
    writer = SummaryWriter(config.OUTPUT)#,config.DATA.DOMAINS[target_idx]))
    domain_num = len(config.DATA.DOMAINS)

    max_acc_path = os.path.join(config.OUTPUT,'distill_max_acc.txt')

    
    max_acc_list=[]
    testAcc_list=[]
    minloss2testacc_list=[]
    acc_ensem_list=[]
    test_ensAcc_list=[]
    test_ens_dpa_Accs = []
    # val_ens_dpa_Accs = []
    test_AllAcc =[[] for _ in config.DATA.DOMAINS]
    test_LastAcc_list=[]
    for target_idx in range(domain_num):
        datasets_train, datasets_val, sources_loader, data_loader_val,data_loader_tar = build_distil_loader(config,target_idx)  
        mixup_fn = build_mixup_fn(config)      
        os.makedirs(os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx],'distill'), exist_ok=True)
        logger = create_logger(output_dir=os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx],'distill') , name=f"{config.DATA.DOMAINS[target_idx]}")
        path = os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx],'distill', "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(config.dump())

        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_model(config)
        model.cuda()
        logger.info(str(model))
        optimizer = build_optimizer(config, model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model, 'flops'):#判断对象是否包含对应的属性
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")


        ensem_model = build_model(config,load_ens=True) 
        ensem_model.cuda()
        model_path = os.path.join(config.ENSEM_PATH,config.DATA.DOMAINS[target_idx], f"{config.DATA.DOMAINS[target_idx]}_model.pth")
        ensem_model.load_state_dict(torch.load(model_path))

        #deepall model
        # dpa_model = build_model(config) 
        # dpa_model.cuda()
        # dpa_folder_path = 'output/PACS/DeepAll_vallia_standardaug/default'
        # dapa_model_path = os.path.join(dpa_folder_path,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
        # dpa_model.load_state_dict(torch.load(dapa_model_path))

        criterion = CrossEntropyLoss()#torch.nn.CrossEntropyLoss()
        criterion_distil = SoftTargetCrossEntropy()
        distill = Distilation(ensem_model,criterion_distil,config.TRAIN.T)


        if config.MODEL.RESUME:
            max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(datasets_val[0])} test images: {acc1:.2f}%")
            if config.EVAL_MODE:
                return

        if config.THROUGHPUT_MODE:
            throughput(data_loader_val, model, logger)
            return


        logger.info("Start training")
        start_time = time.time()




        lr_scheduler = build_scheduler(config, optimizer, len(sources_loader))
        max_accuracy= 0.0
        min_loss = 1e10
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            train_one_epoch(config, model,criterion, distill,sources_loader, optimizer, epoch, mixup_fn, lr_scheduler,logger)
            if  epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                #save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)
                pass 

            acc_sum=0.0

            acc1, loss = validate(config, data_loader_val, model,logger)
            acc_test,loss_test = validate(config, data_loader_tar, model,logger)
            acc_ensem,loss_ensem = validate(config, data_loader_val,ensem_model,logger,EnsModel=True)
            acc_ensem_test,loss_ensem_test = validate(config, data_loader_tar,ensem_model,logger,EnsModel=True)
            # acc_dpa_add_ens,loss_dpa_add_ens = ens_add_deepall_validate(config, data_loader_tar,dpa_model,ensem_model,logger)
            # valacc_dpa_add_ens,valloss_dpa_add_ens = ens_add_deepall_validate(config, data_loader_val,dpa_model,ensem_model,logger)
            writer.add_scalar(f'Val acc/{config.DATA.DOMAINS[target_idx]}',acc1,epoch)
            test_AllAcc[target_idx].append(acc_test)
            writer.add_scalar(f'Val loss/{config.DATA.DOMAINS[target_idx]}',loss,epoch)
            writer.add_scalar(f'Test acc/{config.DATA.DOMAINS[target_idx]}',acc_test,epoch)
            writer.add_scalar(f'Test loss/{config.DATA.DOMAINS[target_idx]}',loss_test,epoch)
            writer.add_scalar(f'Ensemble acc/{config.DATA.DOMAINS[target_idx]}',acc_ensem,epoch)
            writer.add_scalar(f'Ensemble loss/{config.DATA.DOMAINS[target_idx]}',loss_ensem,epoch)
            logger.info(f"Val accuracy&loss : Ensemble:{acc_ensem:.2f}\t {loss_ensem:.4f}\n \
                                                    Distill: {acc1:.2f}\t{loss:.4f}")
            logger.info(f"Test Acc&loss:      Ensemble :{acc_ensem_test:.2f}\t {loss_ensem_test:4f} \n \
                                                    Distill:{acc_test:.2f}\t      {loss_test:.4f}")
            # logger.info(f"DeepAll+Ensemble Test Acc&loss  :{acc_dpa_add_ens:.2f}\t {loss_dpa_add_ens:4f} " )       
            # logger.info(f"DeepAll+Ensemble Val Acc&loss  :{valacc_dpa_add_ens:.2f}\t {valloss_dpa_add_ens:4f} " )  

            model_path = os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx], 'distill',f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")

            if acc1 > max_accuracy:
                if config.TRAIN.MODEL_SELECTION == 'valacc':
                    torch.save(model.state_dict(),model_path)                
                val2testAcc = acc_test
            if loss < min_loss:
                if config.TRAIN.MODEL_SELECTION == 'lossacc':
                    torch.save(model.state_dict(),model_path)            
                loss2testAcc = acc_test
            if config.TRAIN.MODEL_SELECTION == 'last':
                torch.save(model.state_dict(),model_path)
            max_accuracy= max(max_accuracy, acc1)
            min_loss = min(min_loss,loss)
            logger.info(f' Max Val accuracy: {max_accuracy:.2f}  -> Test acc:{val2testAcc:.2f}')


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
        max_acc_list.append(max_accuracy)
        testAcc_list.append(val2testAcc)
        minloss2testacc_list.append(loss2testAcc)
        acc_ensem_list.append(acc_ensem)
        test_ensAcc_list.append(acc_ensem_test)
        # test_ens_dpa_Accs.append(acc_dpa_add_ens)
        # val_ens_dpa_Accs.append(valacc_dpa_add_ens)
        #write acc log
        test_LastAcc_list.append(acc_test)
    first_line = deepcopy(config.DATA.DOMAINS)
    first_line.append('avg\n')
    writeline('',['\n'],max_acc_path) 
    writeline('                ',first_line,max_acc_path) 
    writeline('ValEnsAcc       ',acc_ensem_list,max_acc_path)
    writeline('TestEnsAcc      ',test_ensAcc_list,max_acc_path)
    # writeline('TestEns+DeepAll   ',test_ens_dpa_Accs,max_acc_path)
    # writeline('ValEns+DeepAll   ',val_ens_dpa_Accs,max_acc_path)
    writeline('ValDistAccMax   ',max_acc_list,max_acc_path)
    writeline('Te2ValMaxDistAcc',testAcc_list,max_acc_path)
    writeline('Te2LossMinDisAcc',minloss2testacc_list,max_acc_path)
    writeline('TeLaDistAcc     ',test_LastAcc_list,max_acc_path)
    test_AllAcc = torch.Tensor(test_AllAcc)
    test_mean = test_AllAcc.mean(dim=0)
    for i in range(test_AllAcc.size(1)):
        writer.add_scalar(f'Avg Test Acc',test_mean[i],i)
    writeline('Max Avg Acc&epoch_idx', [torch.max(test_mean,0)[0].item(),torch.max(test_mean,0)[1].item()] , max_acc_path)
        

def writeline(name,varieties,path):
    with open(path, "a") as f:
        f.write(f'{name}\t')
        for variety in varieties:
            if isinstance(variety,str):
                f.write(f'{variety}\t\t')
            else:
                f.write(f'{variety:.2f}\t\t')
        if not isinstance(variety,str):
            f.write(f'{np.mean(varieties):.2f}\n')
        else:
            f.write(f'\n')


def train_one_epoch(config, model,criterion,distill, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,logger):
    model.train()
    domain_num=len(config.DATA.DOMAINS)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()
    if not config.TRAIN.RANDOM_SAMPLER: #uniform sampler
        num_steps = len(list(zip(*data_loader)))
        data_loader = zip(*data_loader)
    else:
        num_steps = len(data_loader)

    #scaler = GradScaler()

    for idx, data in enumerate(data_loader):
        if config.TRAIN.RANDOM_SAMPLER:
            samples = data[0].cuda(non_blocking=True)
            targets = data[1].cuda(non_blocking=True)
            domain_labels=data[2].cuda(non_blocking=True)
            outputs = model(samples)
            loss_cata =criterion(outputs,targets.long())
            if config.DISTILL.IDENTICAL_LOGIT == True:
                loss_distil = distill.get_loss(samples,outputs,domain_labels=domain_labels)
            else:
                loss_distil = distill.get_loss(samples,outputs)
            gamma = config.DISTILL.GAMMA
            loss = (loss_cata + gamma * (config.TRAIN.T**2) * loss_distil)/config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
        else:
            samples=[]
            targets=[]
            for index in range(domain_num-1):
                samples.append(data[index][0].cuda(non_blocking=True))
                targets.append(data[index][1].cuda(non_blocking=True))
            samples = torch.cat(samples)
            targets = torch.cat(targets)
            outputs = model(samples)
            loss_cata =criterion(outputs,targets.long())
            loss_distil = distill.get_loss(samples,outputs)
            gamma = config.DISTILL.GAMMA
            loss = (loss_cata + gamma * (config.TRAIN.T**2) * loss_distil)/config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())        

        if((idx+1)%config.TRAIN.ACCUMULATION_STEPS)==0:
        # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient


        # optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            #acc = accuracy(outputs,targets)
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if DEBUG:
                logger.info(
                    # f'{config.DATA.DOMAINS[target_idx]}\t'
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'             
                    f'mem {memory_used:.0f}MB')   
            tensor_step=epoch*(num_steps//config.PRINT_FREQ)+idx//config.PRINT_FREQ
            #writer.add_scalar(f'Train loss/{config.DATA.DOMAINS[target_idx]}', loss_meter.val,tensor_step)
            #writer.add_scalar(f'Train Acc/{config.DATA.DOMAINS[target_idx]}', acc[0], tensor_step)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model,logger,EnsModel=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    domain_num = len(config.DATA.DOMAINS)
    end = time.time()
    for idx, data in enumerate(data_loader):
        images = data[0].cuda(non_blocking=True)
        target = data[1].cuda(non_blocking=True)
        # compute output
        if not EnsModel:
            output= model(images)
        else :
            inputs =  [deepcopy(images) for  _ in range(domain_num-1)]
            out = model(inputs)
            output = torch.sum(torch.sum(torch.stack(out),0),0)

        # measure accuracy and record loss
        loss = criterion(output, target.long())
        acc1 = accuracy(output, target)
        acc1 = torch.Tensor(acc1)
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return acc1_meter.avg, loss_meter.avg

@torch.no_grad()
def ens_add_deepall_validate(config, data_loader, dpal_model,ens_model,logger,):
    criterion = torch.nn.CrossEntropyLoss()
    dpal_model.eval()
    ens_model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    domain_num = len(config.DATA.DOMAINS)
    end = time.time()
    for idx, data in enumerate(data_loader):
        images = data[0].cuda(non_blocking=True)
        target = data[1].cuda(non_blocking=True)
        # compute deep all output
        dpa_output=torch.softmax( dpal_model(images),-1)
        # compute ensembel output
        inputs_rp =  [deepcopy(images) for  _ in range(domain_num-1)]
        ens_outs = ens_model(inputs_rp)
        ens_out_mean = []
        for index in range(domain_num-1):
            ens_out_mean.append(torch.softmax(ens_outs[index][1]/config.TRAIN.T,-1))
        ens_out_mean = (sum(ens_out_mean)/(domain_num-1))

        #distill cof
        cof_dit = ( config.DISTILL.GAMMA * (config.TRAIN.T**2) )    / (1 + config.DISTILL.GAMMA * (config.TRAIN.T**2) )
        cof_dit = 0.5
        output = (1 - cof_dit) * dpa_output + cof_dit *  ens_out_mean

        # measure accuracy and record loss
        loss = criterion(output, target.long())
        acc1 = accuracy(output, target)
        acc1 = torch.Tensor(acc1)
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return acc1_meter.avg, loss_meter.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    seed = config.SEED 
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)    #used for gpus
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    main(config)
