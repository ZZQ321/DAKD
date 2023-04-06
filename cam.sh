# for method in gradcam hirescam gradcam++ scorecam xgradcam ablationcam eigencam eigengradcam layercam fullgrad
# do
# python gradcam2.py --use-cuda  --aug_smooth --eigen_smooth --method ${method} 
# python gradcam2.py --use-cuda  --eigen_smooth --method ${method} 
# python gradcam2.py --use-cuda  --aug_smooth  --method ${method} 
# python gradcam2.py --use-cuda   --method ${method} 
# done

for method in gradcam fullgrad   xgradcam  gradcam++
do
python gradcam2.py --use-cuda   --method ${method} 
python gradcam2.py --use-cuda  --aug_smooth  --method ${method} 
done


for method in gradcam fullgrad   xgradcam  gradcam++
do
python gradcam2.py --use-cuda  --aug_smooth --eigen_smooth --method ${method} 
python gradcam2.py --use-cuda  --eigen_smooth --method ${method} 
done