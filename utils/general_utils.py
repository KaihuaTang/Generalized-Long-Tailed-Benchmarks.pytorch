######################################
#         Kaihua Tang
######################################
import torch
import numpy as np
import importlib


def count_dataset(train_loader):
    label_freq = {}
    if isinstance(train_loader, list) or isinstance(train_loader, tuple):
        all_labels = train_loader[0].dataset.labels
    else:
        all_labels = train_loader.dataset.labels
    for label in all_labels:
        key = str(label)
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    return label_freq_array


def compute_adjustment(train_loader, tro=1.0):
    """compute the base probabilities"""
    label_freq = {}
    for key in train_loader.dataset.labels:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments


def update(config, algo_config, args):
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.load_dir:
        config['load_dir'] = args.load_dir

    # select the algorithm we use
    if args.train_type:
        config['training_opt']['type'] = args.train_type
        # update algorithm details based on training_type
        algo_info = algo_config[args.train_type]
        config['sampler'] = algo_info['sampler']
        config['num_sampler'] = algo_info['num_sampler'] if 'num_sampler' in algo_info else 1
        config['batch_split'] = algo_info['batch_split'] if 'batch_split' in algo_info else False
        config['testing_opt']['type'] = algo_info['test_type']
        config['training_opt']['loss'] = algo_info['loss_type']
        config['networks']['type'] = algo_info['backbone_type']
        config['classifiers']['type'] = algo_info['classifier_type']
        config['algorithm_opt'] = algo_info['algorithm_opt']
        config['dataset']['rand_aug'] = algo_info['rand_aug'] if 'rand_aug' in algo_info else False

        if 'num_epochs' in algo_info:
            config['training_opt']['num_epochs'] = algo_info['num_epochs']
        if 'batch_size' in algo_info:
            config['training_opt']['batch_size'] = algo_info['batch_size']
        if 'optim_params' in algo_info:
            config['training_opt']['optim_params'] = algo_info['optim_params']
        if 'scheduler' in algo_info:
            config['training_opt']['scheduler'] = algo_info['scheduler']
        if 'scheduler_params' in algo_info:
            config['training_opt']['scheduler_params'] = algo_info['scheduler_params']
        

    # other updates
    if args.lr:
        config['training_opt']['optim_params']['lr'] = args.lr
    if args.testset:
        config['dataset']['testset'] = args.testset
    if args.model_type:
        config['classifiers']['type'] = args.model_type
    if args.loss_type:
        config['training_opt']['loss'] = args.loss_type
    if args.sample_type:
        config['sampler'] = args.sample_type
    if args.rand_aug:
        config['dataset']['rand_aug'] = True
    if args.save_all:
        config['saving_opt']['save_all'] = True
    return config


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_grad(named_parameters):
    """ show grads """
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()
    total_norm = total_norm ** (1. / 2)
    print('---Total norm {:.3f} -----------------'.format(total_norm))
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
    print('-------------------------------', flush=True)
    return total_norm

def print_config(config, logger, head=''):
    for key, val in config.items():
        if isinstance(val, dict):
            logger.info(head + str(key))
            print_config(val, logger, head=head + '   ')
        else:
            logger.info(head + '{} : {}'.format(str(key), str(val)))

class TriggerAction():
    def __init__(self, name):
        self.name = name
        self.action = {}

    def add_action(self, name, func):
        assert str(name) not in self.action
        self.action[str(name)] = func

    def remove_action(self, name):
        assert str(name) in self.action
        del self.action[str(name)]
        assert str(name) not in self.action

    def run_all(self, logger=None):
        for key, func in self.action.items():
            if logger:
                logger.info('trigger {}'.format(key))
            func()


def calculate_recall(prediction, label, split_mask=None):
    recall = (prediction == label).float()
    if split_mask is not None:
        recall = recall[split_mask].mean().item()
    else:
        recall = recall.mean().item()
    return recall


def calculate_precision(prediction, label, num_class, split_mask=None):
    pred_count = torch.zeros(num_class).to(label.device)
    for i in range(num_class):
        pred_count[i] = (prediction == i).float().sum()

    precision = (prediction == label).float() / pred_count[prediction].float() 
    
    if split_mask is not None:
        available_class = len(set(label[split_mask].tolist()))
        precision = precision[split_mask].sum().item() / available_class
    else:
        available_class = len(set(label.tolist()))
        precision = precision.sum().item() / available_class

    return precision


def calculate_f1(recall, precision):
    f1 = 2 * recall * precision / (recall + precision)
    return f1