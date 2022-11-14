import torch
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import seaborn as sns
import pandas as pd
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("white")
palette_dm = sns.color_palette("bright", 3)
palette_cata = sns.color_palette("bright", 7)
'''t-SNE'''

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
import pdb

def tsne_plot(sou_X,sou_y,tar_X,tar_y,target_idx):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501#, n_iter=5000,n_iter_without_progress=1500
                        )
    sou_len = len(sou_X)
    tar_len = len(tar_X)
    X = torch.cat((sou_X,tar_X),0)
    X_tsne = tsne.fit_transform(X)
    cata=['Dog','Elephant','Giraffe','Guitar','Horse','House' ,'Person']
    domain=['artpairing','carton','photo','sketch']
    # data = pd.DataFrame({'X': X, 'y': y})
    # y=torch.LongTensor(y)
    
    db_label = [domain[sou_y[i]] for i in range(len(sou_y))]
    sns.scatterplot(X_tsne[:sou_len,0], X_tsne[:sou_len,1], hue=db_label, legend='full', palette=palette_dm)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(f'dakd_tsne{target_idx}_sou.png')#, bbox_inches='tight')
    plt.cla()

    tar_label = [cata[tar_y[i]] for i in range(len(tar_y))]
    sns.scatterplot(X_tsne[sou_len:,0], X_tsne[sou_len:,1], hue=tar_label, legend='full', palette=palette_cata)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(f'dakd_tsne{target_idx}_tar.png')#, bbox_inches='tight')
    plt.cla()


    # print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    
    # '''嵌入空间可视化'''
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
    #             fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file