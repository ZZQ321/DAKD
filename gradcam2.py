# pip install grad_cam

import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import get_config
from models.build import build_model
import os
import pdb
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both2.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    parser.add_argument('--cfg', type=str, default='configs/PACS_dis.yaml', metavar="FILE", help='path to config file', )



 ########
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',#有传参 use-checkpoint就是true
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', action='store_true', help='If use pretrained parametres')
    parser.add_argument('--gamma',type=float,help='gamma of loss')
    parser.add_argument('--samesub_epoch',type=int,help='')
    parser.add_argument('--ens_tag',type=str,help='')
    parser.add_argument('--dis_tag',type=str,help='')
    parser.add_argument('--model_select',type=str,help='')
    parser.add_argument('--ens_lamda',type=float,help='gamma of loss')
    parser.add_argument('--shared_keys',nargs='+',help='')
    parser.add_argument('--classifier',type=str,help='')
    parser.add_argument('--aug',type=str,help='')
    parser.add_argument('--seed',type=int,help='')
    parser.add_argument('--dis_tar_type',type=str,help='')
    parser.add_argument('--distill',type=str,help='')
 # #######   

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    
    config = get_config(args)

    return args,config


def produce_cam(args,img_path,model,model_name,target_idx,load_ens=False):
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    # args,config = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}


    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    if load_ens:
        target_layers = [model.models[0].layer4]
    else:    
        target_layers = [model.layer4]
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    foldername = 'PACS_' + args.method
    if args.aug_smooth:
        foldername = foldername + 'aug_smooth_'
    if args.eigen_smooth:
        foldername = foldername + 'eigen_smooth'
        
    img_path = img_path.replace('PACS', foldername)
    domain_path = '/'.join(img_path.split('/')[:-1])
    #如果不存在路径img_path，则创建路径
    # filename = "/user/project/demo.txt"
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(domain_path):
        os.makedirs(domain_path)
    
    store_cam = img_path.split('.')[0]+args.method+ model_name+'_cam_'+'{}'.format(target_idx) + '.jpg'

    store_gb = img_path.split('.')[0]+args.method+ model_name+'_gb.jpg'
    store_cam_gb = img_path.split('.')[0]+args.method+ model_name+'_cam_gb.jpg'

    
    cv2.imwrite(store_cam, cam_image)
    # cv2.imwrite( store_gb, gb)
    # cv2.imwrite( store_cam_gb, cam_gb)

if __name__ == '__main__':
   args,config = get_args()
   for target_idx in range(0,3):
    model_erm = build_model(config,load_ens=True)
    #    erm_path = 'output/PACS/DeepAll_vallia_standardaug/default'
    #    erm_path = os.path.join(erm_path,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
    erm_path = 'output/PACS/res18_real_no_mul/default'
    erm_path = os.path.join(erm_path,config.DATA.DOMAINS[target_idx], f"{config.DATA.DOMAINS[target_idx]}_model.pth")
    model_erm.load_state_dict(torch.load(erm_path))

    model_dakd = build_model(config)
    dakd_path = 'output/PACS/res18_real_no_mul/conv#fc_inv_specific'
    dakd_path = os.path.join(dakd_path,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
    model_dakd.load_state_dict(torch.load(dakd_path))

    model_base = build_model(config)
    base_path = 'output/PACS/res18_real_no_mul/noaug'
    base_path = os.path.join(base_path,config.DATA.DOMAINS[target_idx],'distill', f"{config.DATA.DOMAINS[target_idx]}_distilled_model.pth")
    model_base.load_state_dict(torch.load(base_path))

    for domain_name  in ['art_painting','photo','cartoon','sketch']:
    #    for domain_name  in ['cartoon','photo','sketch']:
            domain_path = os.path.join('/home/zzq/data/PACS/',domain_name)
            for class_name in os.listdir(domain_path):
                class_path = os.path.join(domain_path,class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path,img_name)
                    # produce_cam(args,img_path,model_erm,'noji',target_idx=target_idx,load_ens=True)
                    produce_cam(args,img_path,model_base,'base',target_idx=target_idx)
                    produce_cam(args,img_path,model_dakd,'dakd',target_idx=target_idx)
        

#python grad_cam.py --use-cuda --shared_keys conv bn classifier