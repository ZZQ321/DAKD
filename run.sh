

python ensemble.py --cfg configs/Special_OfficeHome_ens_res50.yaml
for gamma in  0.2 1.0
do
python distill.py --cfg configs/Special_OfficeHome_dis_res50.yaml --gamma ${gamma} --tag gamma${gamma}
done

python ensemble.py --cfg configs/Special_OfficeHome_ens.yaml
for gamma in  1.0 0.2 
do
python distill.py --cfg configs/Special_OfficeHome_dis.yaml --gamma ${gamma} --tag gamma${gamma}
done