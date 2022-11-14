# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from copy import deepcopy
import os
from sklearn.utils import shuffle
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.transforms import AutoAugment
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torch.utils.data.dataset import ConcatDataset
import bisect

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from torch.utils.data import random_split
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.utils.data.dataset
import torch.nn
import torchvision
from numpy import arange, var
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
# from .domain_datasets import DomainDataset, Aggregate_DomainDataset
try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


import pdb

class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   
    
    def __len__(self): 
        return len(self.indices)

class CustomConcatDataset(ConcatDataset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, datasets):
        super().__init__(datasets)
        # self.targets = dataset.targets
        # self.classes = dataset.classes
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        x,y = self.datasets[dataset_idx][sample_idx]

        return x,y,dataset_idx

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override
    def __init__(self):
        self.split = False

    def __getitem__(self, index):

        return self.datasets[index]

    def __len__(self):


        return len(self.datasets)

    def min_spl_len(self,istrain):
        min_len = 1e10
        for dataset in self.datasets: 
            if isinstance(dataset,list):
                if istrain:
                    length = len(dataset[0])
                else:
                    length = len(dataset[1])     
                if length<min_len:
                    min_len = length
        if min_len == 1e10:
            NotImplementedError('Only surpport split dataset')
        return min_len
    
    def get_datasets(self):
        datasets_src_tr =[]
        datasets_src_val=[]
        datasets_src = []  
        datasets_tar = []
        for idx in range(len(self)):
            if idx in self.test_envs:
                datasets_tar.append(self.datasets[idx])
            else:
                if self.split:
                    datasets_src_tr.append(self.datasets[idx][0])
                    datasets_src_val.append(self.datasets[idx][1])
                else:
                    datasets_src.append(self.datasets[idx])
        
        if self.split:
            datasets_src = [datasets_src_tr,datasets_src_val]
        return datasets_src,datasets_tar



class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes,test_env,hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        if hparams['split']:
            self.split = True

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            dataset = dataset_transform(images, labels, environments[i])
            if hparams['split'] and i not in test_env:
                length = len(self.datasets[-1])
                train_set,val_set = random_split(dataset,[round(length*0.8),round(length*0.2)])
                self.datasets.append([train_set,val_set])               
            else:
                self.datasets.append(dataset)

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2,test_envs,hparams)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10,test_envs,hparams)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams,img_size=224):
        super().__init__()
        if hparams['split']:
            self.split = True
        self.test_envs = test_envs
        if hparams['split']=='special':
            root_=os.path.join(root,'train')
            environments = [f.name for f in os.scandir(root_) if f.is_dir()]
        else:
            environments = [f.name for f in os.scandir(root) if f.is_dir()]

        environments = sorted(environments)
        # environments = ['photo','art_painting' ,'cartoon' ,'sketch']

        if img_size==224:
            augment_transform =  aug_transform(img_size=224,augment=hparams['aug'])
            transform = aug_transform(img_size=224,augment='noaug')
            transform_src_val =  transform_tar  = transform
            self.input_shape = (3, 224, 224,)

            # dassl Resize(cfg.INPUT.SIZE, interpolation=interp_mode) 
            # Random2DTranslation(cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])
            # Normalize


        elif img_size==32:
            transform_tar = transforms.Compose([
                    transforms.Resize((img_size,img_size),interpolation=_pil_interp('bilinear')),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
            augment_transform = transforms.Compose([
                    transforms.Resize((img_size,img_size),interpolation=_pil_interp('bilinear')),
                    # transforms.CenterCrop((img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
            self.input_shape = (3, 32, 32,)


        self.datasets = []

        for i, environment in enumerate(environments):
            
            if  (i not in test_envs):

                if self.split:
                    if hparams['split'] == 'inclass':
                        path = os.path.join(root, environment)
                        env_dataset = ImageFolder(path)
                        env_dataset=split_dataset_class_wide(env_dataset,hparams['split_rate'],augment_transform,transform_src_val )
                    elif hparams['split'] == 'indomain':

                        path = os.path.join(root, environment)
                        env_dataset = ImageFolder(path)
                        length = len(env_dataset)
                        sr= hparams['split_rate']
                        train_size,validate_size=round(sr*length),round((1-sr)*length)
                        # first param is data set to be saperated, the second is list stating how many sets we want it to be.
                        # origin_data = deepcopy(train_data)
                        shuffle = torch.randperm(length)
                        train_set = CustomSubset(env_dataset, shuffle[:train_size],subset_transform=augment_transform)
                        val_set = CustomSubset(env_dataset, shuffle[train_size:],subset_transform=transform_src_val)
                        # train_set,val_set=torch.utils.data.random_split(env_dataset,[train_size,validate_size])
                        env_dataset = [train_set,val_set]   
                    elif hparams['split'] == 'special':
                        train_path = os.path.join(root,'train',environment)
                        train_set = ImageFolder(train_path,transform=augment_transform)
                        val_path = os.path.join(root,'val',environment)
                        val_set = ImageFolder(val_path,transform=transform_src_val)
                        env_dataset = [train_set,val_set]   
                    else:
                        NotImplementedError('bad split param')
                else:
                    path = os.path.join(root, environment)
                    env_dataset =  ImageFolder(path, transform=augment_transform)          
                    
                                            
            else:
                if hparams['split'] == 'special':
                    path = os.path.join(root,'test',environment)
                else:
                    path = os.path.join(root,environment)
                env_dataset = ImageFolder(path, transform=transform_tar)  
                self.num_classes = len(env_dataset.classes)

            self.datasets.append(env_dataset)

        
        

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    NUM_CLASSES = 5
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams)

class PACS_SPL_VAL(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    NUM_CLASSES = 7
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "pacs_official_split/val/")
        super().__init__(self.dir, test_envs, hparams)

class PACS_SPL_TR(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    NUM_CLASSES = 7
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "pacs_official_split/train/")
        super().__init__(self.dir, test_envs, hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    NUM_CLASSES = 7
    # ENVIRONMENTS = ['photo','art_painting' ,'cartoon' ,'sketch']
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "pacs_official_split/")
        super().__init__(self.dir, test_envs, hparams)

class PACS_NOSPLIT(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    NUM_CLASSES = 7
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    NUM_CLASSES = 345
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams)
   

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    NUM_CLASSES = 65
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    NUM_CLASSES = 10
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    NUM_CLASSES = 7
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams)

class DigitsDG(MultipleDomainDataset):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["MNIST", "MNIST_M", "SVHN", "SYN"]
    NUM_CLASSES = 10
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        img_size =32
        if hparams['split']:
            self.split = True
        self.test_envs = test_envs
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        transform = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=_pil_interp('bilinear')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
        augment_transform = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=_pil_interp('bilinear')),
                # transforms.CenterCrop((img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
        self.input_shape = (3, 32, 32,)


        self.datasets = []

        for i, environment in enumerate(environments):
            train_path = os.path.join(root,environment,'train')
            val_path = os.path.join(root,environment,'val')            
            if  (i not in test_envs):

                if self.split:
                        train_set = ImageFolder(train_path,transform=augment_transform)
                        val_set = ImageFolder(val_path,transform=transform)
                        env_dataset = [train_set,val_set]   
                else:
                    
                    env_dataset =  ImageFolder(train_path,transform=augment_transform) + \
                                   ImageFolder(val_path,transform=augment_transform)         
                    
                                            
            else:
                env_dataset =  ImageFolder(train_path,transform=transform) + \
                                ImageFolder(val_path,transform=transform) 

            self.datasets.append(env_dataset)
        self.num_classes = 10


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams)

def aug_transform(augment='standard',img_size=224):
    if augment == 'auto':
        
        augment_transform =  transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        AutoAugment(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augment == 'rand-m9-mstd0.5-inc1':
        augment_transform = build_transform(is_train=True,img_size=img_size,test_crop=True)
    elif augment == 'standard':
        augment_transform =  transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),#p=0.5
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),#SelfReg:(.4,.4,.4,.5)
                transforms.RandomGrayscale(),#0.1  #FACT don't have it but FACT's follow jigsaw has
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    elif augment == 'db_standard':
            augment_transform = transforms.Compose([    #Domainbed
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif augment == 'noaug':
        augment_transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(img_size), #SelfReg FACT Jigsaw RSC doesn't has it
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        NotImplementedError('augument rule error')
    
    return augment_transform

def build_transform(is_train, img_size,test_crop=False):#config to hparam 
    resize_im = img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(img_size, padding=4)
        return transform

    t = []
    if resize_im:
        if test_crop:
            size = int((256 / 224) * img_size)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp('bicubic')),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(img_size))
        else:
            t.append(
                transforms.Resize((img_size, img_size),
                                  interpolation=_pil_interp('bicubic'))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def split_dataset_class_wide(dataset,ratio,train_transform,val_transform):
    classses_num = len(dataset.classes)
    dataset_len = len(dataset.targets)
    # index = torch.LongTensor([idx for idx in range(dataset_len)])
    index = [idx for idx in range(dataset_len)]
    train_index_all = []
    val_index_all = []
    for target in range(classses_num):
        # target_indicate = (dataset.targets==target)
        # _trace()
        target_index = [i for i in index if dataset.targets[i] == target]
        target_num = len(target_index)
        shuffle = torch.randperm(target_num)
        train_target_num = round(target_num*ratio)
        val_target_num = target_num - train_target_num
        shuffle_train_index = shuffle[:train_target_num]
        shuffle_val_index = shuffle[train_target_num:]
        target_index = torch.LongTensor(target_index)
        train_index = target_index[shuffle_train_index]
        # train_index = [t for i in range(target_num) if ]
        val_index = target_index[shuffle_val_index]
        train_index = train_index.tolist()
        val_index =val_index.tolist()
        train_index_all.extend(train_index)
        val_index_all.extend(val_index)
    train_dataset = CustomSubset(dataset,train_index_all,train_transform)
    val_dataset = CustomSubset(dataset,val_index_all,val_transform)

    return [train_dataset,val_dataset]





if __name__=='__main__':
    root = '/home/zzq/data'
    hparams={'split':False}
    dataset = OfficeHome(root,[0],hparams)
    print(len(dataset))
    print(dataset.split)