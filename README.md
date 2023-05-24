# Welcome to DAKD
## Our DAKD is split into two stage:
## 1- train  multi-expert
## 2- oriented knowledge distillation

change data path in config.py line30:
_C.DATA.DATA_PATH = '/data/path'

## Related command for DAKD

for OfficeHome with Resnet-18 backbone
python ensemble.py --cfg configs/OfficeHome_ens.yaml  
python distill.py --cfg configs/OfficeHome_dis.yaml 

for OfficeHome with Resnet-50 backbone
python ensemble.py --cfg configs/OfficeHome_ens_res50.yaml  
python distill.py --cfg configs/OfficeHome_dis_res50.yaml

for PACS with Resnet-18 backbone
python ensemble.py --cfg configs/PACS_ens.yaml  
python distill.py --cfg configs/PACS_dis.yaml

for PACS with Resnet-50 backbone
python ensemble.py --cfg configs/PACS_ens_res50.yaml  
python distill.py --cfg configs/PACS_dis_res50.yaml


for DomainNet with Resnet-18 backbone
python ensemble.py --cfg configs/SpecialDomainNet_ens.yaml
python distill.py --cfg configs/SpecialDomainNet_dis.yaml

for DomainNet with Resnet-50 backbone
python ensemble.py --cfg configs/SpecialDomainNet_ens_res50.yaml
python distill.py --cfg configs/SpecialDomainNet_dis_res50.yaml