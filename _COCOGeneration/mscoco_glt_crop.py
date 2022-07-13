# Python 2
#from sklearn.externals import joblib
# Python 3
import joblib

import argparse
from PIL import Image, ImageDraw
from io import BytesIO
import json
import joblib
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

# annotation path
ANNOTATION_LT = './coco_intra_lt_inter_lt.jbl'
ANNOTATION_BL = './coco_intra_lt_inter_bl.jbl'
ROOT_PATH = '/data4/'


# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/data4/coco2014/images/', type=str, help='indicate the path of images.')
parser.add_argument('--output_path', default='/data4/coco_glt/', type=str, help='output path.')
parser.add_argument('--seed', default=25, type=int, help='Fix the random seed for reproduction. Default is 25.')
args = parser.parse_args()


def id_to_path(data_path=args.data_path):
    id2path = {}
    subpath = ['val2014', 'train2014']
    for spath in subpath:
        for file in os.listdir(data_path + spath):
            if file.endswith(".jpg"):
                idx = int(file.split('.')[0].split('_')[-1])
                id2path[idx] = os.path.join(data_path, spath, file)
    return id2path

def generate_images_labels(output_img_path, output_ano_path, cocottributes_all):
    annotations = {}
    cat2id = cocottributes_all['cat2id']
    id2cat = {i:cat for cat,i in cat2id.items()}
    annotations['cat2id'] = cat2id
    annotations['id2cat'] = id2cat
    annotations['key2path'] = {}

    train_count_array = get_att_count(cocottributes_all, 'train')
    
    for setname in ['train', 'val', 'test_lt', 'test_bl', 'test_bbl']:
        annotations[setname] = {'label':{}, 'frequency':{}, 'attribute':{}, 'path':{}, 'attribute_score':{}}
        # check validity
        all_keys = list(cocottributes_all[setname]['label'].keys())
        if len(all_keys) == 0:
            print('Skip {}'.format(setname))
            continue
        
        # attribute distribution
        annotations[setname]['attribute_dist'] = get_att_count(cocottributes_all, setname)
        for cat_id in annotations[setname]['attribute_dist'].keys():
            annotations[setname]['attribute_dist'][cat_id] = annotations[setname]['attribute_dist'][cat_id].tolist()
        
        # find attribute threshold
        for coco_attr_key in all_keys:
            cat_id    = cocottributes_all[setname]['label'][coco_attr_key]
            att_array = cocottributes_all[setname]['attribute'][coco_attr_key]
            base_score = normalize_vector(train_count_array[cat_id])
            attr_score = normalize_vector((torch.from_numpy(att_array) > 0.5).float())
            annotations[setname]['attribute_score'][coco_attr_key] = (base_score * attr_score).sum().item()
        att_scores = list(annotations[setname]['attribute_score'].values())
        att_scores.sort(reverse=True)
        attribute_high_mid_thres = att_scores[len(att_scores) // 3]
        attribute_mid_low_thres = att_scores[len(att_scores) // 3 * 2]
        
        for i, coco_attr_key in enumerate(all_keys):
            if (i%1000 == 0):
                print('==== Processing : {}'.format(i/len(all_keys)))
            # generate image
            print_coco_attributes_instance(cocottributes_all, id2path, coco_attr_key, output_img_path.format(coco_attr_key), setname)
            # generate label
            annotations[setname]['label'][coco_attr_key] = cocottributes_all[setname]['label'][coco_attr_key]
            annotations[setname]['path'][coco_attr_key] = output_img_path.format(coco_attr_key)
            annotations[setname]['frequency'][coco_attr_key] = cocottributes_all[setname]['frequency'][coco_attr_key]
            if annotations[setname]['attribute_score'][coco_attr_key] > attribute_high_mid_thres:
                annotations[setname]['attribute'][coco_attr_key] = 0
            elif annotations[setname]['attribute_score'][coco_attr_key] > attribute_mid_low_thres:
                annotations[setname]['attribute'][coco_attr_key] = 1
            else:
                annotations[setname]['attribute'][coco_attr_key] = 2
            
    with open(output_ano_path, 'w') as outfile:
        json.dump(annotations, outfile)
        
        
def normalize_vector(vector):
    output = vector / (vector.sum() + 1e-9)
    return output

        
def get_att_count(cocottributes, setname):
    split_data = cocottributes[setname]
    cat2id = cocottributes['cat2id']
    
    # update array count
    count_array = {}
    for item in set(cat2id.values()):
        count_array[item] = torch.FloatTensor([0 for i in range(len(cocottributes['attributes']))])
    for key in split_data['label'].keys():
        cat_id = split_data['label'][key]
        att_array = split_data['attribute'][key]
        count_array[cat_id] = count_array[cat_id] + (torch.from_numpy(att_array) > 0.5).float()
    
    return count_array


def print_coco_attributes_instance(cocottributes, id2path, coco_attr_id, sname, setname):
    # List of COCO Attributes
    coco_annotation = cocottributes['annotations'][coco_attr_id]
    img_path = id2path[coco_annotation['image_id']]
    img = Image.open(img_path)
    bbox = coco_annotation['bbox']
    
    # crop the object bounding box
    if bbox[2] < 100:
        x1 = max(bbox[0]-50,0)
        x2 = min(bbox[0]+50+bbox[2],img.size[0])
    else:
        x1 = max(bbox[0]-bbox[2]*0.2,0)
        x2 = min(bbox[0]+1.2*bbox[2],img.size[0])
    
    if bbox[3] < 100:
        y1 = max(bbox[1]-50,0)
        y2 = min(bbox[1]+50+bbox[3],img.size[1])
    else:
        y1 = max(bbox[1]-bbox[3]*0.2,0)
        y2 = min(bbox[1]+1.2*bbox[3],img.size[1])
        
    img = img.crop((x1, y1, x2, y2))
    
    # padding the rectangular boxes to square boxes
    w, h = img.size
    pad_size = (max(w,h) - min(w,h))/2
    if w > h:
        img = img.crop((0, -pad_size, w, h+pad_size))
    else:
        img = img.crop((-pad_size, 0, w+pad_size, h))
    
    # save image
    img.save(sname)
    
    
DATA_TYPE = 'coco_bl'
# output path
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
    
    
# generate Train-GLT
if not os.path.exists(os.path.join(args.output_path, 'coco_glt')):
    os.mkdir(os.path.join(args.output_path, 'coco_glt'))

if not os.path.exists(os.path.join(args.output_path, 'coco_glt', 'images')):
    os.mkdir(os.path.join(args.output_path, 'coco_glt', 'images'))

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
output_img_path = os.path.join(args.output_path, 'coco_glt', 'images') + '/{}.jpg'
output_ano_path = os.path.join(args.output_path, 'coco_glt', 'annotation.json')

cocottributes_all = joblib.load(ANNOTATION_LT)

id2path = id_to_path()
generate_images_labels(output_img_path, output_ano_path, cocottributes_all)


# generate Train-CBL
if not os.path.exists(os.path.join(args.output_path, 'coco_cbl')):
    os.mkdir(os.path.join(args.output_path, 'coco_cbl'))

if not os.path.exists(os.path.join(args.output_path, 'coco_cbl', 'images')):
    os.mkdir(os.path.join(args.output_path, 'coco_cbl', 'images'))

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
output_img_path = os.path.join(args.output_path, 'coco_cbl', 'images') + '/{}.jpg'
output_ano_path = os.path.join(args.output_path, 'coco_cbl', 'annotation.json')

cocottributes_all = joblib.load(ANNOTATION_BL)

id2path = id_to_path()
generate_images_labels(output_img_path, output_ano_path, cocottributes_all)
