import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

@torch.no_grad()
def cal_latent_featurs_with_domain_lab(data_loader,model,domain_num):
    features = torch.Tensor([])   
    for idx,data in enumerate(zip(*data_loader)):
        features_batch=[]
        for index in range(domain_num):
            features_batch.append(model.extract_features(data[index][0].cuda(non_blocking=True))[1])
        features_batch_ = torch.stack(features_batch).cpu()

        features = torch.cat((features,features_batch_),dim=1)
    fea_len = features.shape[1] 

    return features,fea_len  


def main(config,args):
# 训练集特征提取
    domain_num = len(config.DATA.DOMAINS)
    for target_idx in range(3,4):
        print('Target Domain:{}'.format(config.DATA.DOMAINS[target_idx]))
        train_loader,test_loader = build_pad_loader(config)  
        os.makedirs(os.path.join(config.OUTPUT,config.DATA.DOMAINS[target_idx]), exist_ok=True)
        model = build_model(config)
        model.cuda()
        model_path = os.path.join(config.DISTILL_PATH,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        train_svm(model,'Baseline',train_loader,test_loader,domain_num)


#MMD
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        db_haparams =  hparams_registry.default_hparams(args.algorithm, 'PACS')
        ctrmodel_path = os.path.join( 'domainbed/output/{}/{}'.format(args.algorithm,target_idx), 'model.pkl')
        ctrmodel_dict = torch.load(ctrmodel_path)['model_dict']
        ctr_algorithms = algorithm_class(input_shape=(3,224,224), num_classes=7, num_domains=3, hparams=db_haparams)
        ctr_algorithms.load_state_dict(ctrmodel_dict)
        ctr_algorithms.cuda()
        train_svm(ctr_algorithms,args.algorithm,train_loader,test_loader,domain_num)

def train_svm(model,model_name,train_loader,test_loader,domain_num):
        domains=['A','C','P','S']
        # dataset_val, dataloaders_src_tr, dataloaders_src_val,dataloader_tar,num_steps_train,num_steps_val = build_loader(config,target_idx)

        # cnn = torch.nn.Sequential(*list(cnn.children())[:-2])
        train_domains_fea,train_fea_len = cal_latent_featurs_with_domain_lab(train_loader,model,domain_num)
        test_domains_fea,test_fea_len = cal_latent_featurs_with_domain_lab(test_loader,model,domain_num)
        train_db_lab = torch.Tensor([0,1]).repeat(train_fea_len,1).permute(1,0).contiguous().view(-1)
        test_db_lab = torch.Tensor([0,1]).repeat(test_fea_len,1).permute(1,0).contiguous().view(-1)
        for  i in range(0,domain_num):
             for j in range(i+1,domain_num):
                train_features = torch.cat((train_domains_fea[i],train_domains_fea[j]))
                test_features = torch.cat((test_domains_fea[i],test_domains_fea[j]))
                  
                # svm = SVC(kernel='linear',probability=True)
                svm = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
                # 训练支持向量机
                svm.fit(train_features, train_db_lab)

                # 测试支持向量机
                criterion = CrossEntropyLoss()
                predicts = svm.predict_proba(test_features)
                predicts = torch.from_numpy(predicts)
                loss = criterion(predicts,test_db_lab.long())
                test_acc = svm.score(test_features, test_db_lab)
                pad = 2 - 4 * (1-test_acc)
                print(model_name+': {}&{} PAD:{:.2f}  Loss: {}'.format(domains[i],domains[j],pad,loss))


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