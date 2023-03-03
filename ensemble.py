# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
#test
from copy import deepcopy
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter

from config import get_config, parse_option
from models import build_model
from data import build_loader,build_mixup_fn
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
#from domain_train import classic_training,classic_test,classic_setting
from tqdm import tqdm
import pdb
DEBUG = 0
def main(config):
    writer = SummaryWriter(config.OUTPUT)#,config.DATA.DOMAINS[target_idx]))
    
    domain_num = len(config.DATA.DOMAINS)

    max_acc_path = os.path.join(config.OUTPUT,'ensemble_max_acc.txt')
    with open(max_acc_path, "w") as f:
        for idx in range(domain_num):
            f.write(f'{config.DATA.DOMAINS[idx]}\t\t')
        f.write('\n')
        
    max_accuracy= np.zeros(domain_num)
    acc_avg_list=[]
    acc_lossmin_list=[]
    for target_idx in range(domain_num):

        dataset_val, dataloaders_src_tr, dataloaders_src_val,dataloader_tar,num_steps_train,num_steps_val = build_loader(config,target_idx)
        mixup_fn = build_mixup_fn(config)
        os.makedirs(os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx]), exist_ok=True)
        logger = create_logger(output_dir=os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx]) , name=f"{config.DATA.DOMAINS[target_idx]}")
        path = os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx], "config.json")
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

        criterion = torch.nn.CrossEntropyLoss()
        criterion_domain = torch.nn.CrossEntropyLoss()


        if config.MODEL.RESUME:
            max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
            acc1, acc5, loss = validate(config, dataloader_tar, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if config.EVAL_MODE:
                return

        if config.THROUGHPUT_MODE:
            throughput(dataloader_tar, model, logger)
            return


        logger.info("Start training")
        start_time = time.time()


        lr_scheduler = build_scheduler(config, optimizer, num_steps_train)
        max_acc_sum = 0.0
        max_acc_split = 0.0
        min_loss_split = 1e10
        # model.pre_train()
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            #data_loader_train.sampler.set_epoch(epoch)
            # if epoch == config.TRAIN.PRE_EPOCH:
            #     model.Add_Mulayer()
            train_one_epoch(config, model, criterion,criterion_domain, dataloaders_src_tr, optimizer, epoch, mixup_fn, lr_scheduler,logger,num_steps_train)
            if  epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                #save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)
                pass 

            acc_sum=0.0
            acc1, loss = validate(config,dataloaders_src_val, model,num_steps_val,logger)
            acc_sig=test(config, dataloader_tar, model,target_idx,logger)
            writer.add_scalar(f'Test acc/{config.DATA.DOMAINS[target_idx]}',acc1,epoch)
            writer.add_scalar(f'Test loss/{config.DATA.DOMAINS[target_idx]}',loss,epoch)
            logger.info(f"Accuracy of the network on the {num_steps_val*config.DATA.BATCH_SIZE*(domain_num-1)} validation images: {acc1:.1f}%")
            model_path = os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx], f"{config.DATA.DOMAINS[target_idx]}_model.pth")
            if acc1 > max_acc_split:
                if config.TRAIN.MODEL_SELECTION == 'valacc':
                    torch.save(model.state_dict(),model_path)
                max_acc_split = max(max_acc_split, acc1)
                acc_sig2valmax = acc_sig
            if loss < min_loss_split:
                if config.TRAIN.MODEL_SELECTION == 'lossacc':
                    torch.save(model.state_dict(),model_path)
                min_loss_split = min(min_loss_split,loss)
                acc_sig2lossmin = acc_sig
            if config.TRAIN.MODEL_SELECTION == 'last':
                torch.save(model.state_dict(),model_path)
            logger.info(f'Max accuracy: {max_acc_split:.2f}%')
        max_accuracy[target_idx]=max_acc_split        
        acc_avg_list.append(acc_sig2valmax)
        acc_lossmin_list.append(acc_sig2lossmin)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

    with open(max_acc_path, "a") as f:
        for idx in range(domain_num):
            f.write(f'{max_accuracy[idx]:.2f},\t\t')
        f.write('\n')

        for acc_avg in acc_avg_list:
            for acc in acc_avg:
                f.write(f'{acc:.2f}\t\t')
            f.write('\n')
        f.write('\n')
        for acc_lossmin in acc_lossmin_list:
            for acc in acc_lossmin:
                f.write(f'{acc:.2f}\t\t')
            f.write('\n')
            


def train_one_epoch(config, model, criterion,criterion_domain, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,logger,num_steps_train):
    model.train()

    num_steps = num_steps_train
    domain_num = len(config.DATA.DOMAINS)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    # loss_d_meter = AverageMeter()
    loss_o_meter = AverageMeter()
    start = time.time()
    end = time.time()
    # for train with no sub
    # if config.TRAIN.SAMESUB_EPOCH > 0:
    #     if epoch == 0:
    #         model.allsource_one_series_layer()
    #     elif epoch ==config.TRAIN.SAMESUB_EPOCH:
    #         model.copy_unshared_para()

    for idx, data in enumerate(  zip(*data_loader)  ):
        samples=[]
        targets=[]
        domain_labels=torch.LongTensor(list(range(domain_num-1)))
        domain_labels=domain_labels.expand(config.DATA.BATCH_SIZE,domain_num-1)
        domain_labels=domain_labels.permute(1,0).contiguous().view(-1).cuda(non_blocking=True)
        for index in range(domain_num-1):
            samples.append(data[index][0].cuda(non_blocking=True))
            targets.append(data[index][1].cuda(non_blocking=True))

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        out = model(samples)
        logits_inv = []
        logits_spe = []
        # domain_logits_list= []
        for index in range(domain_num-1):
            logits_inv.append(out[index][0])
            logits_spe.append(out[index][1])
            # domain_logits_list.append(out[index][2])
        logits_inv = torch.cat(logits_inv)
        logits_spe = torch.cat(logits_spe)
        logits = logits_inv + logits_spe
        # domain_logits = torch.cat(domain_logits_list)
        targets =torch.cat(targets)
        ce_loss = criterion(logits, targets.long())
        # domain_loss = criterion_domain(domain_logits,domain_labels)
        orth_loss = torch.norm(sum(logits_spe*(logits_inv-logits_spe)))
        loss = ce_loss  #+ config.TRAIN.ENSEM_LAMDA*orth_loss
        
        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        # loss_d_meter.update(domain_loss.item(),targets.size(0))
        loss_o_meter.update(orth_loss.item(),targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if DEBUG:
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    # f'Loss_d {loss_d_meter.val:.4f} ({loss_d_meter.avg:.4f})\t'
                    f'Loss_orth {loss_o_meter.val:.4f} ({loss_o_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'             
                    f'mem {memory_used:.0f}MB')   
            #tensor_step=epoch*(num_steps//config.PRINT_FREQ)+idx//config.PRINT_FREQ
            #writer.add_scalar(f'Train loss/{config.DATA.DOMAINS[target_idx]}', loss_meter.val,tensor_step)
            #writer.add_scalar(f'Train Acc/{config.DATA.DOMAINS[target_idx]}', acc[0], tensor_step)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")



@torch.no_grad()
def validate(config, data_loader, model,num_steps_val,logger):
    domain_num = len(config.DATA.DOMAINS)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_d =torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # loss_d_meter = AverageMeter()
    loss_o_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc_d_meter = AverageMeter()
    acc_sig_meter = [AverageMeter() for _ in range(domain_num-1)]

    end = time.time()
    for idx, data in enumerate(zip(*data_loader)):
        samples=[]
        targets=[]
        domain_labels=[]
        for index in range(domain_num-1):
            label_len = len(data[index][0])
            samples.append(data[index][0].cuda(non_blocking=True))
            targets.append(data[index][1].cuda(non_blocking=True))
            domain_labels.append(torch.LongTensor([index]).repeat(label_len).cuda(non_blocking=True))

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        out = model(samples)

        logits_inv = []
        logits_spe = []
        # domain_logits_list= []
        for index in range(domain_num-1):
            logits_inv.append(out[index][0])
            logits_spe.append(out[index][1])
            # domain_logits_list.append(out[index][2])
        logits_inv = torch.cat(logits_inv)
        logits_spe = torch.cat(logits_spe)
        # domain_logits = torch.cat(domain_logits_list)
        #single domain acc
        acc_sig = [0.0 for _ in range(domain_num-1)]
        for souce_idx in range(domain_num -1):
            acc_sig[souce_idx] = accuracy(out[souce_idx][1],targets[souce_idx])[0].item()
            acc_sig_meter[souce_idx].update(acc_sig[souce_idx], targets[souce_idx].size(0))

        #for multi-domain acc    
        targets =torch.cat(targets)
        domain_labels = torch.cat(domain_labels)
        logits = logits_inv + logits_spe
        # measure accuracy and record loss
        loss = criterion(logits, targets.long())
        # loss_d = criterion_d(domain_logits,domain_labels)
        loss_o =torch.norm(sum(logits_spe*(logits_inv-logits_spe),-1))
        acc1 = accuracy(logits, targets)
        # acc_d =accuracy(domain_logits,domain_labels)
        acc1 = torch.Tensor(acc1)
        # acc_d = torch.Tensor(acc_d)
        loss_meter.update(loss.item(), targets.size(0))
        # loss_d_meter.update(loss_d.item(),targets.size(0))
        loss_o_meter.update(loss_o.item(),targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        # acc_d_meter.update(acc_d.item(),domain_labels.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{num_steps_val}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss_ce {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                # f'Loss_d {loss_d_meter.val:.4f} ({loss_d_meter.avg:.4f})\t'
                f'Loss_orth {loss_o_meter.val:.4f} ({loss_o_meter.avg:.4f})\t'
                f'Acc {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                # f'Domain acc {acc_d_meter.val:.3f} ({acc_d_meter.avg:.3f})'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc {acc1_meter.avg:.3f} ')
    return acc1_meter.avg, loss_meter.avg

@torch.no_grad()
def test(config, data_loader, model,target_idx,logger):
    domain_num = len(config.DATA.DOMAINS)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion_d =torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()

    acc_sig_meter = [AverageMeter() for _ in range(domain_num)]

    end = time.time()
    for idx, (samples,targets) in enumerate(data_loader):

        samples=samples.cuda(non_blocking=True)
        targets=targets.cuda(non_blocking=True)
        inputs = [deepcopy(samples) for  _ in range(domain_num-1)]
        out = torch.stack(model(inputs))
        out = torch.sum(out,1)
        acc_sig = [0.0 for _ in range(domain_num-1)]
        for source_idx in range(domain_num-1):
                acc_sig[source_idx] = accuracy(out[source_idx],targets)[0].item()
        acc_sig.insert(target_idx,0.0)

        for source_idx in range(domain_num):
                acc_sig_meter[source_idx].update(acc_sig[source_idx], targets.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    acc_sig_avg = []
    for source_idx in range(domain_num):
        if source_idx!=target_idx:
            acc_sig_avg.append(acc_sig_meter[source_idx].avg)
            logger.info(f'{config.DATA.DOMAINS[source_idx]}->{config.DATA.DOMAINS[target_idx]}:{acc_sig_avg[source_idx]:.2f}\t')
        else:
            acc_sig_avg.append(0.0)
            logger.info(f'{config.DATA.DOMAINS[source_idx]}->{config.DATA.DOMAINS[target_idx]}:{acc_sig_avg[source_idx]:.2f}\t')

    return acc_sig_avg



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
    seed = config.SEED 
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)    #used for gpus
    np.random.seed(seed)
    # cudnn.benchmark = True
    cudnn.deterministic = True
    main(config)
