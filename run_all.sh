for dataset in PACS OfficeHome VLCS Terra 
do
    python ensemble.py --cfg configs/${dataset}_ens.yaml
    for gamma in 1.0
        do
        python distill.py --cfg configs/${dataset}_dis.yaml --gamma ${gamma} --tag gamma_${gamma}
        done 
    
    python ensemble.py --cfg configs/${dataset}_ens_res50.yaml
    for gamma in 1.0 
        do
        python distill.py --cfg configs/${dataset}_dis_res50.yaml --gamma ${gamma} --tag gamma_${gamma}
        done 

done      