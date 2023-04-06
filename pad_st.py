import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC
import numpy as np
from models import build_model
from data import build_pad_loader
import torch.backends.cudnn as cudnn
from config import get_config, parse_option
from torch.nn import CrossEntropyLoss
from random import randint
import os
import pdb

from domainbed.algorithms import MMD
from domainbed import hparams_registry
from domainbed import algorithms
PI=[i for i in range(11)]    # *10了
@torch.no_grad()
def cal_latent_featurs_with_domain_lab(data_loader,model,domain_num,pi,target_idx):
    model.eval()
    features = torch.Tensor([])
    dlabls = torch.Tensor([])
    for idx,data in enumerate(zip(*data_loader)):
        samples=[]
        targets=[]
        B = data[0][0].shape[0]
        
        for index in range(domain_num):
            samples.append(data[index][0].cuda(non_blocking=True))
            targets.append(data[index][1].cuda(non_blocking=True))
        tar_samples = samples.pop(target_idx)
        num_per_sou =  [int(p*B) for p in pi]
        sou_samples = [samples[i][:num_per_sou[i]]  for i in range(domain_num-1)]
        sou_samples = torch.cat(sou_samples)
        sou_len = sou_samples.shape[0]
        tar_len = B
        sou_dblabel = torch.Tensor([0]).repeat(sou_len)
        tar_dblabel = torch.Tensor([1]).repeat(tar_len)
        domain_labels = torch.cat([sou_dblabel,tar_dblabel])
        samples = torch.cat([sou_samples,tar_samples])
        _,features_batch = model.extract_features(samples)
        features = torch.cat([features,features_batch.cpu()])
        dlabls = torch.cat([dlabls,domain_labels])
    return features, dlabls    

def serach_best_pi_for_min_st_div(train_loader,test_loader,model,model_name,domain_num,target_idx):
    #遍历所有的每个元素不同的包含3个元素的组合
    pi_combs=[]
    for pi_0 in PI:
        sum = pi_0
        if sum < 10:
            left = 10 - sum
            pi_left = [i for i in range(left+1)]
            for pi_1 in pi_left:
                pi_2 = 10 - (pi_0+pi_1)
                pi_combs.append([pi_0*0.1,pi_1*0.1,pi_2*0.1])

        else:
            pi_1=pi_2=0
            pi_combs.append([pi_0*0.1,pi_1*0.1,pi_2*0.1])
    pads = []
    for pi_comb in pi_combs:
        train_features,train_labels = cal_latent_featurs_with_domain_lab(train_loader,model,domain_num,pi_comb,target_idx)
        test_features, test_labels = cal_latent_featurs_with_domain_lab(test_loader,model,domain_num,pi_comb,target_idx)
        # 定义支持向量机
        svm = SVC(kernel='linear',probability=True)
        # 训练支持向量机
        svm.fit(train_features, train_labels)
        # 测试支持向量机
        test_acc = svm.score(test_features, test_labels)
        epsilen = 1 - test_acc
        pad = 2 * (1 - 2*epsilen)
        pads.append(pad)
        print('{} -> {}'.format(pi_comb,pad))
    
    pad_min = min(pads)
    pdb.set_trace()
    min_idx = pads.index(pad_min)
    pi_min = pi_combs[min_idx]
    print(model_name + '   Min ST distance for Target {}  ->   Min:{:2f}  PI:{}'.format( config.DATA.DOMAINS[target_idx], pad_min, pi_min))

    

        

def main(config,args):
# 训练集特征提取
    domain_num = len(config.DATA.DOMAINS)
    for target_idx in range(domain_num):
        train_loader,test_loader = build_pad_loader(config)  
        os.makedirs(os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx]), exist_ok=True)
        model = build_model(config)
        model.cuda()
        model_path = os.path.join(config.DISTILL_PATH,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        serach_best_pi_for_min_st_div(train_loader,test_loader,model,'dakd',domain_num,target_idx)
        print('\n')


#MMD    
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        db_haparams =  hparams_registry.default_hparams(args.algorithm, 'PACS')
        ctrmodel_path = os.path.join( 'domainbed/output/{}/{}'.format(args.algorithm,target_idx), 'model.pkl')
        ctrmodel_dict = torch.load(ctrmodel_path)['model_dict']
        ctr_algorithms = algorithm_class(input_shape=(3,224,224), num_classes=7, num_domains=3, hparams=db_haparams)
        ctr_algorithms.load_state_dict(ctrmodel_dict)
        ctr_algorithms.cuda()
        serach_best_pi_for_min_st_div(train_loader,test_loader,model,args.algorithm,domain_num,target_idx)
        print('\n\n')

def train_svm(model,model_name,train_loader,test_loader,domain_num,target_idx):

        # dataset_val, dataloaders_src_tr, dataloaders_src_val,dataloader_tar,num_steps_train,num_steps_val = build_loader(config,target_idx)

        # cnn = torch.nn.Sequential(*list(cnn.children())[:-2])

        train_features,train_labels = cal_latent_featurs_with_domain_lab(train_loader,model,domain_num,target_idx)
        # 测试集特征提取
        test_features, test_labels = cal_latent_featurs_with_domain_lab(test_loader,model,domain_num,target_idx)

        # 定义支持向量机
        svm = SVC(kernel='linear',probability=True)
        # 训练支持向量机
        svm.fit(train_features, train_labels)

        # 测试支持向量机
        criterion = CrossEntropyLoss()
        predicts = svm.predict_proba(test_features)
        predicts = torch.from_numpy(predicts)
        loss = criterion(predicts,test_labels.long())
        print(model_name+':Target{}测试误差：{:.2f}'.format(target_idx,loss))
        test_acc = svm.score(test_features, test_labels)
        print(model_name+':Target{}测试准确率：{:.4f}'.format(target_idx,test_acc))


def cumulative_sum(numbers):
    """
    将数量列表转化为累计数量的函数

    参数:
    numbers (list): 包含数量的列表

    返回值:
    list: 包含累计数量的列表
    """
    cumulative_numbers = []  # 用于存储累计数量的列表
    total = 0  # 初始累计数量为 0

    for number in numbers:
        total += number  # 将当前数量加到累计数量上
        cumulative_numbers.append(total)  # 将累计数量添加到结果列表中

    return cumulative_numbers


if __name__ == '__main__':
    args, config = parse_option()
    seed = config.SEED 
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)    #used for gpus
    np.random.seed(seed)
    # cudnn.benchmark = True
    cudnn.deterministic = True
    main(config,args)

    #python pad.py --cfg configs/PACS_dis.yaml