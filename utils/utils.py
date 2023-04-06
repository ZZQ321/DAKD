import torch
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import seaborn as sns
import pandas as pd
sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.set(rc={'figure.figsize':(23.4,16.54)})
sns.set_style("white")
palette_sou_dm = sns.color_palette("bright", 3)
palette_dm = sns.color_palette("bright", 4)
palette_cata = sns.color_palette("bright", 7)
'''t-SNE'''

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
import pdb

def tsne_plot_togeter(model_name,sou_X,sou_cata,sou_db,sou_exp,tar_X,tar_cata,tar_exp,target_idx):
    os.makedirs(model_name,exist_ok=True)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501#, n_iter=5000,n_iter_without_progress=1500
                        )
    sou_len = len(sou_X)
    tar_len = len(tar_X)

    soutar_cata = torch.cat((sou_cata,tar_cata))

    tar_db_label = torch.LongTensor([target_idx]).repeat(tar_len)
    sou_db_label = torch.where(sou_db<target_idx,sou_db,sou_db+1)
    sou_tar_db_label = torch.cat((sou_db_label,tar_db_label),0)
    
    if sou_exp is not None:
        sou_exp = torch.where(sou_exp<target_idx,sou_exp,sou_exp+1)
        tar_exp = torch.where(tar_exp<target_idx,tar_exp,tar_exp+1)
        sou_tar_exp = torch.cat((sou_exp,tar_exp),0)
        expert=['A','C','P','S']
        sou_exp_names = [expert[sou_exp[i]] for i in range(len(sou_exp))]
        tar_exp_names = [expert[tar_exp[i]] for i in range(len(tar_exp))]
        sou_tar_exp_names = sou_exp_names + tar_exp_names
    else:
        sou_exp_names = None
        tar_exp_names = None
        sou_tar_exp_names = None
    X = torch.cat((sou_X,tar_X),0)
    # X = torch.nn.functional.normalize(X,dim=-1)
    print(torch.where(torch.isinf(X) | torch.isnan(X),1,0).sum().sum())
    X_tsne = tsne.fit_transform(X)
  
    cata=['Dog','Elephant','Giraffe','Guitar','Horse','House' ,'Person']
    domain=['artpairing','carton','photo','sketch']
    
    # data = pd.DataFrame({'X': X, 'y': y})
    # y=torch.LongTensor(y)

# source domain
#   db    
    sou_db_label_names = [domain[sou_db_label[i]] for i in range(len(sou_db_label))]
    sns.scatterplot(X_tsne[:sou_len,0], X_tsne[:sou_len,1], hue=sou_db_label_names, legend='full', palette=palette_sou_dm,style=sou_exp_names)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_sou_db.png')#, bbox_inches='tight')
    plt.cla()
#   cata    
    sou_cata_names = [cata[sou_cata[i]] for i in range(len(sou_cata))]
    sns.scatterplot(X_tsne[:sou_len,0], X_tsne[:sou_len,1], hue=sou_cata_names, legend='full', palette=palette_cata,style=sou_exp_names)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_sou_cata.png')#, bbox_inches='tight')
    plt.cla()


# all domain
#db
    soutar_db_label_names = [domain[sou_tar_db_label[i]] for i in range(len(sou_tar_db_label))]
    sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=soutar_db_label_names, legend='auto', palette=palette_dm,style=sou_tar_exp_names)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_soutar_db.png')#, bbox_inches='tight')
    plt.cla()
#cata
    soutar_cata_names = [cata[soutar_cata[i]] for i in range(len(soutar_cata))]
    sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=soutar_cata_names, legend='auto', palette=palette_cata,style=sou_tar_exp_names)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_soutar_cata.png')#, bbox_inches='tight')
    plt.cla()

#target doamin
    tar_label = [cata[tar_cata[i]] for i in range(len(tar_cata))]
    sns.scatterplot(X_tsne[sou_len:,0], X_tsne[sou_len:,1], hue=tar_label, legend='auto', palette=palette_cata)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_tar_cata.png')#, bbox_inches='tight')
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

def tsne_plot_separate(model_name,sou_X,sou_cata,sou_db,sou_exp,tar_X,tar_cata,tar_exp,target_idx):
    os.makedirs(model_name,exist_ok=True)
    tsne_sou = manifold.TSNE(n_components=2, init='pca', random_state=501#, n_iter=5000,n_iter_without_progress=1500
                        )
    tsne_tar = manifold.TSNE(n_components=2, init='pca', random_state=501#, n_iter=5000,n_iter_without_progress=1500
                        )
    sou_len = len(sou_X)
    tar_len = len(tar_X)

    soutar_cata = torch.cat((sou_cata,tar_cata))

    tar_db_label = torch.LongTensor([target_idx]).repeat(tar_len)
    sou_db_label = torch.where(sou_db<target_idx,sou_db,sou_db+1)
    sou_tar_db_label = torch.cat((sou_db_label,tar_db_label),0)
    
    if sou_exp is not None:
        sou_exp = torch.where(sou_exp<target_idx,sou_exp,sou_exp+1)
        tar_exp = torch.where(tar_exp<target_idx,tar_exp,tar_exp+1)
        sou_tar_exp = torch.cat((sou_exp,tar_exp),0)
        expert=['A','C','P','S']
        sou_exp_names = [expert[sou_exp[i]] for i in range(len(sou_exp))]
        tar_exp_names = [expert[tar_exp[i]] for i in range(len(tar_exp))]
        sou_tar_exp_names = sou_exp_names + tar_exp_names
    else:
        sou_exp_names = None
        tar_exp_names = None
        sou_tar_exp_names = None
    # X = torch.cat((sou_X,tar_X),0)
    # X = torch.nn.functional.normalize(X,dim=-1)
    X_sou_tsne= tsne_sou.fit_transform(sou_X)
    X_tar_tsne= tsne_sou.fit_transform(tar_X)
    # print(torch.where(torch.isinf(X) | torch.isnan(X),1,0).sum().sum())

  
    cata=['Dog','Elephant','Giraffe','Guitar','Horse','House' ,'Person']
    domain=['art painting','cartoon','photo','sketch']
    
    # data = pd.DataFrame({'X': X, 'y': y})
    # y=torch.LongTensor(y)

# source domain
#   db    
    sou_db_label_names = [domain[sou_db_label[i]] for i in range(len(sou_db_label))]
    sns.scatterplot(X_sou_tsne[:,0], X_sou_tsne[:,1], hue=sou_db_label_names, legend='full', palette=palette_sou_dm,style=sou_exp_names)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_sou_db.png')#, bbox_inches='tight')
    plt.cla()
#   cata    
    sou_cata_names = [cata[sou_cata[i]] for i in range(len(sou_cata))]
    sns.scatterplot(X_sou_tsne[:,0], X_sou_tsne[:,1], hue=sou_cata_names, legend='full', palette=palette_cata,style=sou_exp_names)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_sou_cata.png')#, bbox_inches='tight')
    plt.cla()



#target doamin
    tar_label = [cata[tar_cata[i]] for i in range(len(tar_cata))]
    sns.scatterplot(X_tar_tsne[:,0], X_tar_tsne[:,1], hue=tar_label, legend='auto', palette=palette_cata)  
    plt.show()
    # path = os.path.join(save_path,'tsne.png')
    plt.savefig(model_name + f'/dist_tsne{target_idx}_tar_cata.png')#, bbox_inches='tight')
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