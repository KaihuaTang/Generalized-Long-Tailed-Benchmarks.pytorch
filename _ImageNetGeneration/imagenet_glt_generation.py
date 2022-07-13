from PIL import Image
import json
import os
import random
import argparse


from sklearn.cluster import KMeans

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset


LT_DIST_PATH = './long-tail-distribution.pytorch'
TRAIN_LT_PATH = './imagenet_sup_intra_lt_inter_lt.json'    # output annotation file: both classes and attributes of training set are long-tailed
TRAIN_KBL_PATH = './imagenet_sup_intra_lt_inter_bl.json'   # output annotation file: classes of training set are balanced, but pretext attributes are still iid sampled, i.e., long-tailed

# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/data4/imagenet/ILSVRC/Data/CLS-LOC/train/', type=str, help='indicate the path of train data for ImageNet.')
parser.add_argument('--num_cluster', default=6, type=int, help='Number of clusters, ie, pretext attributes.')
parser.add_argument('--seed', default=25, type=int, help='Fix the random seed for reproduction. Default is 25.')
args = parser.parse_args()


# ============================================================================
# fix random seed
print('=====> Using fixed random seed: ' + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ============================================================================
# Data loader
class ImageNetExtract(Dataset):
    def __init__(self, index, data_path):
        super(ImageNetExtract, self).__init__()
        categories = os.listdir(data_path)
        names = os.listdir(os.path.join(data_path, categories[index]))
        self.category = categories[index]
        self.images = [os.path.join(data_path, categories[index], name) for name in names]
        self.transform = transforms.Compose([
                            transforms.Resize(260),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = self.images[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            sample = self.transform(sample)
        return sample, index
    
def get_loader(index, batch_size=256):
    dataset = ImageNetExtract(index, args.data_path)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    return loader, dataset


# ============================================================================
# Get Features for Each Class
def get_outputs(model, index):
    # show reconstructed image
    indexes = []
    features = []
    out_paths = []
    model.eval()
    val_loader, val_data = get_loader(index=index, batch_size=32)
    img_paths = val_data.images
    category = val_data.category
    
    with torch.no_grad():
        for images, ids in val_loader:
            images = images.cuda()
            z = model(images)
            indexes.append(ids.view(-1).cpu())
            features.append(z.cpu())
        indexes = torch.cat(indexes, dim=0).numpy()
        features = torch.cat(features, dim=0).numpy()
    
    for idx in list(indexes):
        out_paths.append(img_paths[idx])
    return features, out_paths, category


# ============================================================================
# Apply Clustering
def get_cluster_label(features, out_paths, category, clusterlabel_dict, NUM_CLUSTER=6):
    clusterlabel_dict[category] = {}
    k_means = KMeans(n_clusters=NUM_CLUSTER).fit(features)
    for path, label in zip(out_paths, list(k_means.labels_)):
        if label in clusterlabel_dict[category]:
            clusterlabel_dict[category][label].append(path)
        else:
            clusterlabel_dict[category][label] = [path,]
    return clusterlabel_dict


# ============================================================================
# change the label of clusters in a decreasing order
def cluster_ranking(clusterlabel_dict):
    new_data = {}
    idx2label = {}
    label_num = []
    count_img = 0
    for key, val in clusterlabel_dict.items():
        num_cat_img = sum([len(item) for item in val.values()])
        label_num.append((key, num_cat_img))
        count_img += num_cat_img
        cluster_paths = list(val.values())
        cluster_paths.sort(key=len, reverse=True)
        new_data[key] = {i : [] for i in range(3)}
        for i, paths in enumerate(cluster_paths):
            new_data[key][i//2] += paths
    # sort category by the number of images
    sorted_labels = sorted(label_num, reverse=True, key=lambda item: item[1])
    # sorted labels
    for i, item in enumerate(sorted_labels):
        idx2label[str(i)] = item[0]
        
    new_data = {str(i) : new_data[idx2label[str(i)]] for i in range(len(idx2label))}
    print('===== Total Number of Images is {} ====='.format(count_img))
    return new_data, idx2label


# ============================================================================
# generate GLT splits
def generate_train_val_test(cls_data, VAL_SIZE=10, TEST_SIZE=20, NUM_CAT=1000, NUM_ATT=3, intra_type='lt', inter_type='bl'):
    trainset =         {'label':{}, 'frequency':{}, 'attribute':{}}
    valset =           {'label':{}, 'frequency':{}, 'attribute':{}}
    testset_lt =       {'label':{}, 'frequency':{}, 'attribute':{}}
    testset_bl =       {'label':{}, 'frequency':{}, 'attribute':{}}
    testset_bbl =      {'label':{}, 'frequency':{}, 'attribute':{}}
    
    # intra distribution
    intra_dist = [7,2,1]
    print('Intra-Distribution is ' + str(intra_dist))
    print_total_img(cls_data)
    
    #################################################
    # TEST BBL: generate test set that has the balanced
    # distribution for both category and attribute
    #################################################
    cls_data, testset_bbl =  generate_balanced_test(cls_data, testset_bbl, TEST_SIZE)
    print_total_img(cls_data)
    
    
    #################################################
    # TEST BL: generate test set that only has the balanced
    # class distribution
    #################################################
    if intra_type == 'lt' and inter_type == 'lt':
        cls_data, testset_bl = generate_intra_lt_test(cls_data, testset_bl, intra_dist, TEST_SIZE, NUM_ATT)
    elif intra_type == 'lt' and inter_type == 'bl':
        cls_data, testset_bl = generate_intra_lt_test(cls_data, testset_bl, intra_dist, TEST_SIZE, NUM_ATT)
    else:
        raise ValueError('Wrong Combination of Distribution Types')
    print_total_img(cls_data)
    

    #################################################
    # Generate Imbalanced Dataset
    #################################################
    if intra_type == 'lt' and inter_type == 'lt':
        cls_data = get_long_tailed_data(cls_data, intra_dist, imb_type='predefined')
    elif intra_type == 'lt' and inter_type == 'bl':
        cls_data = get_long_tailed_data(cls_data, intra_dist, imb_type='bl')
    else:
        raise ValueError('Wrong Combination of Distribution Types')
    print_total_img(cls_data)
    
        
    #################################################
    # TEST LT: generate test set that has the same
    # distribution with long-tailed train set
    #################################################
    if intra_type == 'lt' and inter_type == 'lt':
        cls_data, testset_lt = generate_iid_set(cls_data, testset_lt, TEST_SIZE, NUM_ATT)
    print_total_img(cls_data)
    
    
    #################################################
    # VAL: generate validation set, the distribution 
    # of val set should be the same as train set
    #################################################
    if intra_type == 'lt' and inter_type == 'lt':
        cls_data, valset = generate_iid_set(cls_data, valset, VAL_SIZE, NUM_ATT)
    elif intra_type == 'lt' and inter_type == 'bl':
        cls_data, valset = generate_intra_lt_test(cls_data, valset, intra_dist, VAL_SIZE, NUM_ATT)
    else:
        raise ValueError('Wrong Combination of Distribution Types')
    print_total_img(cls_data)
        
        
    #################################################
    # TRAIN
    #################################################
    for key, val in cls_data.items():
        for c_key, c_val in val.items():
            for item in c_val:
                trainset['label'][item] = key
                trainset['frequency'][item] = get_frequency_label(key)
                trainset['attribute'][item] = c_key
    print('====== Total Num of Train Set is {} ========'.format(len(trainset['label'])))
    
    return trainset, valset, testset_lt, testset_bl, testset_bbl
        

def get_frequency_label(input_label):
    if int(input_label) < 400:
        return 0
    elif int(input_label) < 850:
        return 1
    else:
        return 2
    
    
def get_long_tailed_data(cls_data, intra_dist, imb_type='exp', imb_factor=0.05, print_table=True):
    # get long_tailed category list
    img_max = 1800
    cls_sizes = [sum([len(c_val) for c_key, c_val in val.items()]) for key, val in cls_data.items()]
    cls_num = len(cls_sizes)
    img_num_per_cls = []

    if imb_type == 'predefined':
        lt_dist = torch.load(LT_DIST_PATH)
        # check validity
        for i in range(len(lt_dist) - 1):
            assert lt_dist[i] >= lt_dist[i+1]
        # set distribution
        img_num_per_cls = lt_dist
    elif imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(min(int(num), cls_sizes[cls_idx]))
    elif imb_type == 'step':
        for cls_idx in range(cls_num):
            if cls_idx < int(cls_num * 0.2):
                img_num_per_cls.append(cls_sizes[cls_idx])
            elif cls_idx < int(cls_num * 0.4):
                img_num_per_cls.append(min(cls_sizes[cls_idx], img_max * 0.5))
            else:
                img_num_per_cls.append(min(cls_sizes[cls_idx], img_max * 0.05))
    elif imb_type == 'bl':
        for cls_idx in range(cls_num):
            img_num_per_cls.append(min(145, cls_sizes[cls_idx]))
    else:
        raise ValueError('Wrong Type')
        
    
    # sampling long-tailed dataset
    att_dist = torch.FloatTensor(intra_dist)
    
    for cls_idx, (key, val) in enumerate(cls_data.items()):
        num_imgs = img_num_per_cls[cls_idx]
        num_atts = ((att_dist / att_dist.sum()) * num_imgs).long().tolist()
 
        for att_idx, (c_key, c_val) in enumerate(val.items()):
            att_len = min(num_atts[att_idx], len(c_val))
            del c_val[att_len:]
    
    return cls_data

def generate_balanced_test(cls_data, input_set, VAL_SIZE):
    # both attribute and category are balanced
    for key, val in cls_data.items():
        # get the size for each attribute cluster
        att_size = VAL_SIZE
        for c_key, c_val in val.items():
            att_size = min(att_size, len(c_val)//4)
        
        # update inputset
        for c_key, c_val in val.items():
            for _ in range(att_size):
                item = c_val.pop(0)
                input_set['label'][item] = key
                input_set['frequency'][item] = get_frequency_label(key)
                input_set['attribute'][item] = c_key
    
    print('====== Total Num of Both Balanced Test Set is {} ========'.format(len(input_set['label'])))
    return cls_data, input_set


def generate_intra_lt_test(cls_data, input_set, intra_dist, VAL_SIZE, NUM_ATT):
    # attribute is iid
    # category is balanced
    TOTAL_SIZE = VAL_SIZE * NUM_ATT
    multi_weight = int(TOTAL_SIZE / sum(intra_dist))
    
    for key, val in cls_data.items():
        for i, (c_key, c_val) in enumerate(val.items()):
            att_size = intra_dist[i] * multi_weight
            for _ in range(att_size):
                item = c_val.pop(0)
                input_set['label'][item] = key
                input_set['frequency'][item] = get_frequency_label(key)
                input_set['attribute'][item] = c_key
    
    print('====== Total Num of Half Balanced Test Set is {} ========'.format(len(input_set['label'])))
    return cls_data, input_set


def generate_iid_set(cls_data, input_set, VAL_SIZE, NUM_ATT):
    # attribute is long-tailed (iid)
    # category is long-tailed (iid)
    num_cls = len(cls_data)
    TOTAL_SIZE = num_cls * VAL_SIZE * NUM_ATT
    
    container = []
    for key, val in cls_data.items():
        for c_key, c_val in val.items():
            for item in c_val:
                container.append( ((key, c_key), item) )
    
    # update inputset
    for i in range(TOTAL_SIZE):
        idx = random.randint(0, len(container)-1)
        item = container.pop(idx)
        key = item[0][0]
        c_key = item[0][1]
        c_val = item[1]
        input_set['label'][c_val] = key
        input_set['frequency'][c_val] = get_frequency_label(key)
        input_set['attribute'][c_val] = c_key
        
    # update new_cls_data
    new_cls_data = {}
    for key, val in cls_data.items():
        new_cls_data[key] = {i:[] for i in range(NUM_ATT)}
    for item in container:
        key = item[0][0]
        c_key = item[0][1]
        c_val = item[1]
        new_cls_data[key][c_key].append(c_val)
    
    print('====== Total Num of IID Set is {} ========'.format(len(input_set['label'])))
    return new_cls_data, input_set
    
def print_total_img(cls_data):
    count = 0
    for key, val in cls_data.items():
        for c_key, c_val in val.items():
            count += len(c_val)
    print('NOTE: The current size of remaining data is {}'.format(count))
    
def save_output(trainset, valset, testset_lt, testset_bl, testset_bbl, idx2label, output_path):
    output_dict = {}
    
    id2cat = idx2label
    cat2id = {cat:i for i, cat in id2cat.items()}
   
    output_dict['cat2id'] = cat2id
    output_dict['id2cat'] = id2cat
    output_dict['train'] = trainset
    output_dict['val'] = valset
    output_dict['test_lt'] = testset_lt
    output_dict['test_bl'] = testset_bl
    output_dict['test_bbl'] = testset_bbl
    
    with open(output_path, 'w') as outfile:
        json.dump(output_dict, outfile)





# ============================================================================
# Main
def main(args):
    # load model
    model = torchvision.models.resnet50(pretrained=True).cuda()
    # Remove last FC layer, because we want features rather than logits
    model.fc = nn.ReLU()
    # generate pretext attribute labels by clustering
    clusterlabel_dict = {}

    for index in range(len(os.listdir(args.data_path))):
        print('======== Extracting {} category ==========='.format(index))
        features, out_paths, category = get_outputs(model, index)
        clusterlabel_dict = get_cluster_label(features, out_paths, category, clusterlabel_dict, NUM_CLUSTER=args.num_cluster)


    # Train-LT and corresponding (Test-LT + Test-KBL + Test-GBL)
    cls_data, idx2label = cluster_ranking(clusterlabel_dict)
    trainset, valset, testset_lt, testset_bl, testset_bbl = generate_train_val_test(cls_data, intra_type='lt', inter_type='lt')
    save_output(trainset, valset, testset_lt, testset_bl, testset_bbl, idx2label, TRAIN_LT_PATH)


    # Train-KLT and corresponding (Test-LT + Test-KBL + Test-GBL)
    cls_data, idx2label = cluster_ranking(clusterlabel_dict)
    trainset, valset, testset_lt, testset_bl, testset_bbl = generate_train_val_test(cls_data, intra_type='lt', inter_type='bl')
    save_output(trainset, valset, testset_lt, testset_bl, testset_bbl, idx2label, TRAIN_KBL_PATH)



if __name__ == '__main__':
    main(args)