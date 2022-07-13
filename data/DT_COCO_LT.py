######################################
#         Kaihua Tang
######################################


import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from randaugment import RandAugment

class COCO_LT(data.Dataset):
    def __init__(self, phase, data_path, anno_path, testset, rgb_mean, rgb_std, rand_aug, output_path, logger):
        super(COCO_LT, self).__init__()
        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase
        if phase == 'train':
            full_phase = 'train'
        elif phase == 'test':
            full_phase = testset
        else:
            full_phase = phase
        logger.info('====== The Current Split is : {}'.format(full_phase))
        self.logger = logger

        self.dataset_info = {}
        self.phase = phase
        self.rand_aug = rand_aug
        self.data_path = data_path

        self.annotations = json.load(open(anno_path))
        self.data = self.annotations[full_phase]
        self.transform = self.get_data_transform(phase, rgb_mean, rgb_std)
        
        # load dataset category info
        logger.info('=====> Load dataset category info')
        self.id2cat, self.cat2id = self.annotations['id2cat'], self.annotations['cat2id']

        # load all image info
        logger.info('=====> Load image info')
        self.img_paths, self.labels, self.attributes, self.frequencies = self.load_img_info()
        
        # save dataset info
        logger.info('=====> Save dataset info')
        self.dataset_info['cat2id'] = self.cat2id
        self.dataset_info['id2cat'] = self.id2cat
        self.save_dataset_info(output_path)


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]
        rarity = self.frequencies[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # intra-class attribute SHOULD NOT be used during training
        if self.phase != 'train':
            attribute = self.attributes[index]
            return sample, label, rarity, attribute, index
        else:
            return sample, label, rarity, index


    #######################################
    #  Load image info
    #######################################
    def load_img_info(self):
        img_paths = []
        labels = []
        attributes = []
        frequencies = []

        for key, label in self.data['label'].items():
            img_paths.append(self.data['path'][key])
            labels.append(int(label))
            frequencies.append(int(self.data['frequency'][key]))

            # intra-class attribute SHOULD NOT be used in training
            if self.phase != 'train':
                att_label = int(self.data['attribute'][key])
                attributes.append(att_label)
               
        # save dataset info
        self.dataset_info['img_paths'] = img_paths
        self.dataset_info['labels'] = labels
        self.dataset_info['attributes'] = attributes
        self.dataset_info['frequencies'] = frequencies

        return img_paths, labels, attributes, frequencies
 

    #######################################
    #  Save dataset info
    #######################################
    def save_dataset_info(self, output_path):

        with open(os.path.join(output_path, 'dataset_info_{}.json'.format(self.phase)), 'w') as f:
            json.dump(self.dataset_info, f)

        del self.dataset_info


    #######################################
    #  transform
    #######################################
    def get_data_transform(self, phase, rgb_mean, rgb_std):
        transform_info = {
            'rgb_mean': rgb_mean,
            'rgb_std':  rgb_std,
        }

        if phase == 'train':
            if self.rand_aug:
                self.logger.info('============= Using Rand Augmentation in Dataset ===========')
                trans = transforms.Compose([
                            transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            RandAugment(),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                        ])
                transform_info['operations'] = ['RandomResizedCrop(112, scale=(0.5, 1.0)),', 'RandomHorizontalFlip()', 
                                            'RandAugment()', 'ToTensor()', 'Normalize(rgb_mean, rgb_std)']
            else:
                self.logger.info('============= Using normal transforms in Dataset ===========')
                trans = transforms.Compose([
                            transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                        ])
                transform_info['operations'] = ['RandomResizedCrop(112, scale=(0.5, 1.0)),', 'RandomHorizontalFlip()', 
                                            'ToTensor()', 'Normalize(rgb_mean, rgb_std)']
        else:
            trans = transforms.Compose([
                            transforms.Resize(128),
                            transforms.CenterCrop(112),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                        ])
            transform_info['operations'] = ['Resize(128)', 'CenterCrop(112)', 'ToTensor()', 'Normalize(rgb_mean, rgb_std)']
        
        # save dataset info
        self.dataset_info['transform_info'] = transform_info

        return trans