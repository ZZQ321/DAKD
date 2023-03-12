

python ensemble.py --cfg configs/PACS_ens_res50.yaml
for gamma in 1.0
do
python distill.py --cfg configs/PACS_dis_res50.yaml --gamma ${gamma}
done