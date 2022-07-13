from PIL import Image
import json
import os
import random
import argparse
import joblib

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset


TRAIN_LT_PATH = './coco_intra_lt_inter_lt.jbl'    # output annotation file: both classes and attributes of training set are long-tailed
TRAIN_KBL_PATH = './coco_intra_lt_inter_bl.jbl'   # output annotation file: classes of training set are balanced, but pretext attributes are still iid sampled, i.e., long-tailed

# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/data4/coco2014/images/', type=str, help='indicate the path of images.')
parser.add_argument('--anno_path', default='/data4/coco2014/annotations/', type=str, help='indicate the path of annotations.')
parser.add_argument('--attribute_path', default='./cocottributes_py3.jbl', type=str, help='indicate the path of coco attributes.')
parser.add_argument('--seed', default=25, type=int, help='Fix the random seed for reproduction. Default is 25.')
args = parser.parse_args()


def load_coco(root_path=args.anno_path):
    # Load COCO Annotations in val2014 & train2014
    coco_data = {'images':[], 'annotations':[]}
    with open(os.path.join(root_path, 'instances_train2014.json'), 'r') as f:
        train2014 = json.load(f)
    with open(os.path.join(root_path, 'instances_val2014.json'), 'r') as f:
        val2014 = json.load(f)
    coco_data['categories'] = train2014['categories']
    coco_data['images'] += train2014['images']
    coco_data['images'] += val2014['images']
    coco_data['annotations'] += train2014['annotations']
    coco_data['annotations'] += val2014['annotations']
    return coco_data


def get_statistics(cocottributes, ann_vecs, coco_data):
    att_statistics = {}
    cat_statistics = {}
    key2cats = {}
    attr_details = sorted(cocottributes['attributes'], key=lambda x:x['id'])
    attr_names = [item['name'] for item in attr_details]
    obj_id2annos = {ann['id'] : ann for ann in coco_data['annotations']}
    cat_id2cats = {c['id'] : c['name'] for c in coco_data['categories']}
    
    for key in list(ann_vecs.keys()):
        instance_attrs = ann_vecs[key]
        # category statistics
        coco_obj_id = cocottributes['patch_id_to_ann_id'][key]
        coco_annotation = obj_id2annos[coco_obj_id]
        cat = cat_id2cats[coco_annotation['category_id']]
        key2cats[key] = cat
        cat_statistics[cat] = cat_statistics.get(cat, 0) + 1

        # attribute statistics
        pos_attrs = [a for att_id, a in enumerate(attr_names) if instance_attrs[att_id] > 0.5]
        for att in pos_attrs:
            att_statistics[att] = att_statistics.get(att, 0) + 1

    return att_statistics, cat_statistics, key2cats


def generate_train_val_test(cocottributes, key2cats, TEST_SIZE=100, VAL_SIZE=50, intra_type='lt', inter_type='lt'):
    ann_vecs = cocottributes['ann_vecs']
    # init data splits
    trainset =         {'label':{}, 'frequency':{}, 'attribute':{}}
    valset =           {'label':{}, 'frequency':{}, 'attribute':{}}
    testset_lt =       {'label':{}, 'frequency':{}, 'attribute':{}}
    testset_bl =       {'label':{}, 'frequency':{}, 'attribute':{}}
    testset_bbl =      {'label':{}, 'frequency':{}, 'attribute':{}}
    
    NUM_CAT = len(cat_statistics)
    NUM_ATT = len(att_statistics)
    print('number of category: ', NUM_CAT)
    print('number of attribute: ', NUM_ATT)
    
    # index keys by category
    cat_keys = {cat : {} for cat in set(key2cats.values())}
    for key, val in ann_vecs.items():
        cat = key2cats[key]
        cat_keys[cat][key] = val
    # generate frequent label
    CAT2FRQ, CAT2ID = generate_freq_label(cat_keys)
    print('Data Size After {} is {}'.format('Init', len(ann_vecs)))
    print('Data Size After {} is {}'.format('Init', sum([len(val) for key, val in cat_keys.items()])))
    
    #################################################
    # TEST BBL: generate test set that has the balanced
    # distribution for both category and attribute
    #################################################
    cat_keys, ann_vecs, testset_bbl = generate_balanced_test(cat_keys, ann_vecs, testset_bbl, TEST_SIZE, NUM_ATT, CAT2FRQ, CAT2ID)
    print('Data Size After {} is {}'.format('Test-BBL', len(ann_vecs)))
    print('Data Size After {} is {}'.format('Test-BBL', sum([len(val) for key, val in cat_keys.items()])))
    
    
    #######################################################
    # TEST BL: generate test set that only has the balanced
    # class distribution
    #######################################################
    cat_keys, ann_vecs, testset_bl = generate_intra_lt_test(cat_keys, ann_vecs, testset_bl, TEST_SIZE, CAT2FRQ, CAT2ID)
    print('Data Size After {} is {}'.format('Test-BL', len(ann_vecs)))
    print('Data Size After {} is {}'.format('Test-BL', sum([len(val) for key, val in cat_keys.items()])))
    
    
    #################################################
    # Generate Imbalanced Dataset
    #################################################
    if intra_type == 'lt' and inter_type == 'lt':
        trainset = generate_data(cat_keys, ann_vecs, trainset, CAT2FRQ, CAT2ID, imb_type='exp')
    elif intra_type == 'lt' and inter_type == 'bl':
        trainset = generate_data(cat_keys, ann_vecs, trainset, CAT2FRQ, CAT2ID, imb_type='bl')
    else:
        raise ValueError('Wrong Combination of Distribution Types')
    print('Data Size After {} is {}'.format('Selection', len(trainset['label'])))
    
    
    #################################################
    # TEST LT: generate test set that has the same
    # distribution with long-tailed train set
    #################################################
    if intra_type == 'lt' and inter_type == 'lt':
        testset_lt = generate_iid_set(trainset, testset_lt, TEST_SIZE * NUM_CAT)
    print('Data Size After {} is {}'.format('Selection', len(trainset['label'])))
    
    
    #################################################
    # VAL: generate validation set, the distribution 
    # of val set should be the same as train set
    #################################################
    valset = generate_iid_set(trainset, valset, VAL_SIZE * NUM_CAT)
    print('Data Size After {} is {}'.format('Selection', len(trainset['label'])))
    
    return trainset, valset, testset_lt, testset_bl, testset_bbl, CAT2ID




# useful functions

def generate_freq_label(cat_keys):
    cls_sizes_with_cats = [(len(val), key) for key, val in cat_keys.items()]
    cls_sizes_with_cats.sort(key=lambda x: x[0], reverse=True)
    print('Class Size with Cat: ', str(cls_sizes_with_cats))
    # freq label and label index
    cat2frq = {}
    cat2id = {}
    for i, item in enumerate(cls_sizes_with_cats):
        # label index
        cat2id[item[1]] = i
        # freq label
        if i <= 10:
            cat2frq[item[1]] = 0
        elif i <= 22:
            cat2frq[item[1]] = 1
        else:
            cat2frq[item[1]] = 2
    return cat2frq, cat2id


def normalize_vector(vector):
    output = vector / (vector.sum() + 1e-9)
    return output


def generate_balanced_test(cat_keys, ann_vecs, outputset, VAL_SIZE, NUM_ATT, CAT2FRQ, CAT2ID):
    for i, (c_key, c_val) in enumerate(cat_keys.items()):
        test_dist = np.array([0.0 for _ in range(NUM_ATT)])
        print('===== Processing: {} ====='.format(i/len(cat_keys)))
        cat_count = 0
        while(cat_count<VAL_SIZE):
            min_std = 999999999.9
            min_key = None
            min_val = None
            for key, val in c_val.items():
                val = (val > 0.5).astype(np.float32)
                if val.sum() == 0:
                    continue
                tmp_dist = (test_dist + val)
                if normalize_vector(tmp_dist).std() < min_std:
                    min_std = normalize_vector(tmp_dist).std()
                    min_key = key
                    min_val = val
            outputset['label'][min_key] = CAT2ID[c_key]
            outputset['frequency'][min_key] = CAT2FRQ[c_key]
            outputset['attribute'][min_key] = min_val
            test_dist += min_val
            cat_count += 1
            del ann_vecs[min_key]
            del c_val[min_key]
    return cat_keys, ann_vecs, outputset



def generate_intra_lt_test(cat_keys, ann_vecs, outputset, VAL_SIZE, CAT2FRQ, CAT2ID):
    for c_key, c_val in cat_keys.items():
        cat_count = 0
        while(cat_count < VAL_SIZE):
            idx = random.randint(0, len(c_val)-1)
            key = list(c_val.keys())[idx]
            outputset['label'][key] = CAT2ID[c_key]
            outputset['frequency'][key] = CAT2FRQ[c_key]
            outputset['attribute'][key] = c_val[key]
            cat_count += 1
            del ann_vecs[key]
            del c_val[key]
    return cat_keys, ann_vecs, outputset



def generate_data(cat_keys, ann_vecs, outputset, CAT2FRQ, CAT2ID, imb_type='bl', imb_factor=0.005):
    cls_sizes_with_cats = [(len(val), key) for key, val in cat_keys.items()]
    cls_sizes_with_cats.sort(key=lambda x: x[0], reverse=True)
    cls_sizes = [item[0] for item in cls_sizes_with_cats]
    max_size, min_size = max(cls_sizes), min(cls_sizes)
    print('Max/min class sizes are {} / {}'.format(max_size, min_size))
    cls_num = len(cls_sizes)
    print('Num of class is {}'.format(cls_num))

    img_num_per_cls = []
    if imb_type == 'bl':
        img_num_per_cls = [min_size] * cls_num
    elif imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = max_size * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(min(int(num), cls_sizes[cls_idx]))
    else:
        print('Wrong imbalance type')
        
    for i, item in enumerate(cls_sizes_with_cats):
        c_key = item[1]
        c_val = cat_keys[c_key]
        cat_count = 0
        THIS_SIZE = img_num_per_cls[i]
        while(cat_count < THIS_SIZE):
            idx = random.randint(0, len(c_val)-1)
            key = list(c_val.keys())[idx]
            outputset['label'][key] = CAT2ID[c_key]
            outputset['frequency'][key] = CAT2FRQ[c_key]
            outputset['attribute'][key] = c_val[key]
            cat_count += 1
            del ann_vecs[key]
            del c_val[key]
        
    return outputset


def generate_iid_set(inputset, outputset, TOTAL_SIZE):
    total_count = 0
    while(total_count < TOTAL_SIZE):
        idx = random.randint(0, len(inputset['label'])-1)
        key = list(inputset['label'].keys())[idx]
        outputset['label'][key] = inputset['label'][key]
        outputset['frequency'][key] = inputset['frequency'][key]
        outputset['attribute'][key] = inputset['attribute'][key]
        total_count += 1
        # remove selected items
        del inputset['label'][key]
        del inputset['frequency'][key]
        del inputset['attribute'][key]
    return outputset



def save_output(cocottributes, trainset, valset, testset_lt, testset_bl, testset_bbl, CAT2ID, coco_data, output_path):
    output_dict = {}
    output_dict.update(cocottributes)
    
    print('Train size is : {}'.format(len(trainset['label'])))
    print('Val size is : {}'.format(len(valset['label'])))
    print('Test-LT size is : {}'.format(len(testset_lt['label'])))
    print('Test-BL size is : {}'.format(len(testset_bl['label'])))
    print('Test-BBL size is : {}'.format(len(testset_bbl['label'])))
    
    output_dict['train'] = trainset
    output_dict['val'] = valset
    output_dict['test_lt'] = testset_lt
    output_dict['test_bl'] = testset_bl
    output_dict['test_bbl'] = testset_bbl
    output_dict['cat2id'] = CAT2ID

    annid2attid = {}
    attanns = {}
    for att_id, ann_id in cocottributes['patch_id_to_ann_id'].items():
        annid2attid[ann_id] = att_id
    for ann in coco_data['annotations']:
        if ann['id'] in annid2attid:
            attanns[annid2attid[ann['id']]] = ann
        
    output_dict['annotations'] = attanns

    
    joblib.dump(output_dict, output_path, compress=True)
    
    


# generate class-wise balanced, intra-class long-tailed training set and corresponding test splits
TEST_SIZE = 100
VAL_SIZE = 50

# load raw annotations
cocottributes = joblib.load(args.attribute_path)
coco_data = load_coco()

# get statistics
att_statistics, cat_statistics, key2cats = get_statistics(cocottributes, cocottributes['ann_vecs'], coco_data)
# generate splits
trainset, valset, testset_lt, testset_bl, testset_bbl, CAT2ID = generate_train_val_test(cocottributes, key2cats, TEST_SIZE=TEST_SIZE, VAL_SIZE=VAL_SIZE, intra_type='lt', inter_type='bl')
save_output(cocottributes, trainset, valset, testset_lt, testset_bl, testset_bbl, CAT2ID, coco_data, TRAIN_KBL_PATH)



# generate class-wise long-tailed, intra-class long-tailed training set and corresponding test splits
TEST_SIZE = 200
VAL_SIZE = 100

# load raw annotations
cocottributes = joblib.load(args.attribute_path)
coco_data = load_coco()

# get statistics
att_statistics, cat_statistics, key2cats = get_statistics(cocottributes, cocottributes['ann_vecs'], coco_data)
# generate splits
trainset, valset, testset_lt, testset_bl, testset_bbl, CAT2ID = generate_train_val_test(cocottributes, key2cats, TEST_SIZE=TEST_SIZE, VAL_SIZE=VAL_SIZE, intra_type='lt', inter_type='lt')
save_output(cocottributes, trainset, valset, testset_lt, testset_bl, testset_bbl, CAT2ID, coco_data, TRAIN_LT_PATH)