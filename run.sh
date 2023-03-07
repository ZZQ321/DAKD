python ensemble.py --cfg configs/PACS_ens.yaml
for gamma in 1.0
do
python distill.py --cfg configs/PACS_dis.yaml --gamma ${gamma}
done

python ensemble.py --cfg configs/PACS_ens.yaml
for gamma in 1.0
do
python distill.py --cfg configs/PACS_dis.yaml --gamma ${gamma}
done