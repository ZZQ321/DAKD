# Our method is split into two stage:
# 1- train  multi-expert
# 2- oriented knowledge distillation

# Related command for DAKD

# for OfficeHome with Resnet-18 backbone
python ensemble.py --cfg configs/OfficeHome_ens.yaml  
python distill.py --cfg configs/OfficeHome_dis.yaml 

# for OfficeHome with Resnet-50 backbone
python ensemble.py --cfg configs/OfficeHome_ens_res50.yaml  
python distill.py --cfg configs/OfficeHome_dis_res50.yaml

# for PACS with Resnet-18 backbone
python ensemble.py --cfg configs/PACS_ens.yaml  
python distill.py --cfg configs/PACS_dis.yaml

# for PACS with Resnet-50 backbone
python ensemble.py --cfg configs/PACS_ens_res50.yaml  
python distill.py --cfg configs/PACS_dis_res50.yaml


# for VLCS with Resnet-18 backbone
python ensemble.py --cfg configs/VLCS_ens.yaml  
python distill.py --cfg configs/VLCS_dis.yaml