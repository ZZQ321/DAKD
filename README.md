# Welcome to DAKD
## Our DAKD is split into two stage:
1.train  multi-expert

2.oriented knowledge distillation

## Set data dir
change data path in config.py line30:
_C.DATA.DATA_PATH = '/data/path'

## Related command

### evalution on OfficeHome using Resnet-18
python ensemble.py --cfg configs/OfficeHome_ens.yaml  

python distill.py --cfg configs/OfficeHome_dis.yaml 

### evalution on OfficeHome using Resnet-50 
python ensemble.py --cfg configs/OfficeHome_ens_res50.yaml  

python distill.py --cfg configs/OfficeHome_dis_res50.yaml

### evalution on PACS using Resnet-18 
python ensemble.py --cfg configs/PACS_ens.yaml  

python distill.py --cfg configs/PACS_dis.yaml

### evalution on PACS using Resnet-50 
python ensemble.py --cfg configs/PACS_ens_res50.yaml 
 
python distill.py --cfg configs/PACS_dis_res50.yaml


### evalution on DomainNet using Resnet-18 
python ensemble.py --cfg configs/SpecialDomainNet_ens.yaml
python distill.py --cfg configs/SpecialDomainNet_dis.yaml

### evalution on DomainNet using Resnet-50 
python ensemble.py --cfg configs/SpecialDomainNet_ens_res50.yaml
python distill.py --cfg configs/SpecialDomainNet_dis_res50.yaml