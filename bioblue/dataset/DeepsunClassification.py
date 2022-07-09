

from email.mime import base
from tokenize import group
from bioblue.dataset.transform.pipelines import Compose
import collections
from functools import partial

from torch.utils.data import Dataset
import os
from hydra.utils import call, instantiate
from pathlib import Path
import skimage.io as io
import json

import numpy as np

import traceback
import time



class DeepsunMaskedClassificationDataset(Dataset):

    def __init__(
        self, root_dir, partition, dtypes, classes, classification='Zurich' , transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        print(dtypes)

        self.transforms = transforms
        self.root_dir = Path(root_dir)

        self.main_dtype = dtypes[0]
        self.mask_dtype = dtypes[1]

        # print(dtypes)
        # print('self.main_dtype',self.main_dtype)
        # print('self.mask_dtype',self.mask_dtype)
        
        self.json_file = self.root_dir / (self.mask_dtype + '.json') 
        self.partition_dict = None

        with open(self.json_file, 'r') as f:
            self.partition_dict = json.load(f)[partition]

        assert len(dtypes) == 2, "dtypes > 2: DeepsunMaskedClassificationDataset can use only one type of mask."
        

        assert (classification == 'Zurich') or (classification == 'McIntosh')

        self.classification = classification

        self.classes_mapper = {c: i for i,c in enumerate(classes)}

        self.files = {}
        for i, bn in enumerate(sorted(list(self.partition_dict.keys()))):
            cur = {}
            image_basename = bn + '.FTS'
            image_filename = self.root_dir / self.main_dtype / image_basename

            mask_basename = bn + '.png'
            mask_filename = self.root_dir / self.mask_dtype / mask_basename

            cur["name"] = bn
            cur[self.main_dtype] = image_filename
            cur[self.mask_dtype] = mask_filename

            self.files[bn] = cur

        # num_groups = 0
        # self.groups = []
        self.groups = {}
        for k,v in self.partition_dict.items():
            for i, g in enumerate(v['groups']):
                k2 = k+'_'+str(i)

                dataset_g = {   
                                "solar_angle": v['angle'],
                                "deltashapeX":v['deltashapeX'],
                                "deltashapeY":v['deltashapeY'],
                                "group_info": g,
                            }

                # self.groups.append({k2: dataset_g})
                self.groups[k2] = dataset_g

            # num_groups += len(v['groups'])
        # self.dataset_length = num_groups
        self.dataset_length = len(list(self.groups.keys()))



    def __len__(self) -> int:
        # raise NotImplementedError
        return self.dataset_length
        # return 10
    
    def __getitem__(self, index: int, do_transform=True):

        sample = {} # dictionnary with 'image', 'class', 'angular_excentricity', 'centroid_lat'

        # basename = self.files[index]["name"]
        k = sorted(list(self.groups.keys()))[index]
        basename = k.split('_')[0]


        # image_out_dict = self.partition_dict[basename]
        group_dict = self.groups[k]

        img_name = self.files[basename][self.main_dtype] # path of FITS file
        mask_name = self.files[basename][self.mask_dtype]

        sample['image'] = (io.imread(img_name)).astype(float) 
        sample['mask'] = io.imread(mask_name).astype(float)

        sample['solar_angle'] = group_dict['solar_angle']
        sample['deltashapeX'] = group_dict['deltashapeX']
        sample['deltashapeY'] = group_dict['deltashapeY']
        
        sample['angular_excentricity'] = np.array([group_dict['group_info']["angular_excentricity_deg"]])
        sample['centroid_px'] = np.array(group_dict['group_info']["centroid_px"])
        sample['centroid_Lat'] = np.array([group_dict['group_info']["centroid_Lat"]])

        sample['class'] = np.array([self.classes_mapper[group_dict['group_info'][self.classification]]])

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        # print([ (key,type(sample[key])) for key in list(sample.keys())])

        return sample
        


class DeepsunMaskedClassificationDatasetV2(Dataset):

    def __init__(
        self, root_dir, partition, dtypes, classes, classification='Zurich' , transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        print(dtypes)

        self.transforms = transforms
        self.root_dir = Path(root_dir)

        self.main_dtype = dtypes[0]
        self.mask_dtype = dtypes[1]

        # print(dtypes)
        # print('self.main_dtype',self.main_dtype)
        # print('self.mask_dtype',self.mask_dtype)
        
        self.json_file = self.root_dir / (self.mask_dtype + '.json') 
        self.partition_dict = None

        with open(self.json_file, 'r') as f:
            self.partition_dict = json.load(f)[partition]

        assert len(dtypes) == 2, "dtypes > 2: DeepsunMaskedClassificationDataset can use only one type of mask."
        

        assert (classification == 'Zurich') or (classification == 'McIntosh')

        self.classification = classification

        self.classes_mapper = {c: i for i,c in enumerate(classes)}

        self.files = {}
        for i, bn in enumerate(sorted(list(self.partition_dict.keys()))):
            bn = bn.split('_')[0]
            # print(bn)
            cur = {}
            image_basename = bn + '.FTS'
            image_filename = self.root_dir / self.main_dtype / image_basename

            mask_basename = bn + '.png'
            mask_filename = self.root_dir / self.mask_dtype / mask_basename

            cur["name"] = bn
            cur[self.main_dtype] = image_filename
            cur[self.mask_dtype] = mask_filename

            self.files[bn] = cur

        # print(self.files)

        self.groups = self.partition_dict

        self.dataset_length = len(list(self.groups.keys()))



    def __len__(self) -> int:
        # raise NotImplementedError
        # print(self.dataset_length)
        return self.dataset_length
        # return 10
    
    def __getitem__(self, index: int, do_transform=True):

        st = time.time()

        sample = {} # dictionnary with 'image', 'class', 'angular_excentricity', 'centroid_lat'

        # basename = self.files[index]["name"]
        k = sorted(list(self.groups.keys()))[index]
        # print(k)
        basename = k.split('_')[0]


        # image_out_dict = self.partition_dict[basename]
        group_dict = self.groups[k]

        # print(group_dict)

        img_name = self.files[basename][self.main_dtype] # path of FITS file
        mask_name = self.files[basename][self.mask_dtype]

        # print(img_name)

        sample['image'] = (io.imread(img_name)).astype(float) 
        sample['mask'] = io.imread(mask_name).astype(float)

        sample['solar_angle'] = group_dict['angle']
        sample['deltashapeX'] = group_dict['deltashapeX']
        sample['deltashapeY'] = group_dict['deltashapeY']
        
        sample['angular_excentricity'] = np.array([group_dict["angular_excentricity_deg"]])
        sample['centroid_px'] = np.array(group_dict["centroid_px"])
        sample['centroid_Lat'] = np.array([group_dict["centroid_Lat"]])

        sample['class'] = np.array([self.classes_mapper[group_dict[self.classification]]])

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        # print([ (key,type(sample[key])) for key in list(sample.keys())])

        et = time.time()
        # print(f'DeepsunMaskedClassificationDatasetV2 getitem time: {et-st} seconds')

        return sample
        