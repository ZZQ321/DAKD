python distill.py --cfg configs/PACS_dis.yaml  --shared_keys conv --distill avg --ens_tag conv_avg --tag conv_avg

for expert in   none conv#fc_inv#bn bn fc_inv fc_inv#bn conv#bn
do
    for distill in  specific avg
    do
        python ensemble.py --cfg configs/PACS_ens.yaml --shared_keys ${expert} --distill ${distill} --ens_tag ${expert}_${distill} --tag ${expert}_${distill} 
        python distill.py --cfg configs/PACS_dis.yaml  --shared_keys ${expert} --distill ${distill} --ens_tag ${expert}_${distill} --tag ${expert}_${distill}
    done

done

for expert in conv#fc_inv conv none conv#fc_inv#bn bn fc_inv fc_inv#bn conv#bn
do
    for distill in  specific avg
    do
        python ensemble.py --cfg configs/PACS_ens_res50.yaml --shared_keys ${expert} --distill ${distill} --ens_tag ${expert}_${distill} --tag ${expert}_${distill} 
        python distill.py --cfg configs/PACS_dis_res50.yaml  --shared_keys ${expert} --distill ${distill} --ens_tag ${expert}_${distill} --tag ${expert}_${distill}
    done

done