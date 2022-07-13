######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F 

import utils.general_utils as utils
from data.dataloader import get_loader
from utils.checkpoint_utils import Checkpoint
from utils.training_utils import *
from utils.test_loader import test_loader

class train_bbn():
    def __init__(self, args, config, logger, eval=False):
        # ============================================================================
        # create model
        logger.info('=====> Model construction from: ' + str(config['networks']['type']))
        model_type = config['networks']['type']
        model_file = config['networks'][model_type]['def_file']
        model_args = config['networks'][model_type]['params']
        logger.info('=====> Classifier construction from: ' + str(config['classifiers']['type']))
        classifier_type = config['classifiers']['type']
        classifier_file = config['classifiers'][classifier_type]['def_file']
        classifier_args = config['classifiers'][classifier_type]['params']
        model = utils.source_import(model_file).create_model(**model_args)
        classifier = utils.source_import(classifier_file).create_model(**classifier_args)

        model = nn.DataParallel(model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

        # other initialization
        self.config = config
        self.logger = logger
        self.model = model
        self.classifier = classifier
        self.optimizer = create_optimizer(model, classifier, logger, config)
        self.scheduler = create_scheduler(self.optimizer, logger, config)
        self.eval = eval
        self.training_opt = config['training_opt']

        self.checkpoint = Checkpoint(config)

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', config['dataset']['testset'], logger)

        # get loss
        self.loss_fc = create_loss(logger, config, self.train_loader)

        # set eval
        if self.eval:
            test_func = test_loader(config)
            self.testing = test_func(config, logger, model, classifier, val=True)


    def calculate_reverse_instance_weight(self, dataloader):
        # counting frequency
        label_freq = {}
        for key in dataloader.dataset.labels:
            label_freq[key] = label_freq.get(key, 0) + 1
        label_freq = dict(sorted(label_freq.items()))
        label_freq_array = torch.FloatTensor(list(label_freq.values()))
        reverse_class_weight = label_freq_array.max() / label_freq_array
        # generate reverse weight
        reverse_instance_weight = torch.zeros(len(dataloader.dataset)).fill_(1.0)
        for i, label in enumerate(dataloader.dataset.labels):
            reverse_instance_weight[i] = reverse_class_weight[label] / (label_freq_array[label] + 1e-9)
        return reverse_instance_weight


    def run(self):
        # Start Training
        self.logger.info('=====> Start BBN Training')

        # preprocess for each epoch
        env1_loader, env2_loader = self.train_loader
        assert len(env1_loader) == len(env2_loader)
        total_batch = len(env1_loader)
        total_image = len(env1_loader.dataset)

        # set dataloader distribution
        instance_normal_weight = torch.zeros(total_image).fill_(1.0)
        env1_loader.sampler.set_parameter(instance_normal_weight)  # conventional distribution
        instance_reverse_weight = self.calculate_reverse_instance_weight(env1_loader)
        env2_loader.sampler.set_parameter(instance_reverse_weight) # reverse distribution

        # run epoch
        num_epoch = self.training_opt['num_epochs']
        for epoch in range(num_epoch):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))

            for step, ((inputs1, labels1, _, indexs1), (inputs2, labels2, _, indexs2)) in enumerate(zip(env1_loader, env2_loader)):
                iter_info_print = {}

                self.optimizer.zero_grad()

                # additional inputs
                inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()
                labels1, labels2 = labels1.cuda(), labels2.cuda()

                feature1 = self.model(inputs1, feature_cb=True)
                feature2 = self.model(inputs2, feature_rb=True)

                l = 1 - ((epoch - 1) / num_epoch) ** 2  # parabolic decay

                mixed_feature = 2 * torch.cat((l * feature1, (1-l) * feature2), dim=1)

                predictions = self.classifier(mixed_feature)
                
                # calculate loss
                loss = l * self.loss_fc(predictions, labels1) + (1 - l) * self.loss_fc(predictions, labels2)
                iter_info_print = {'BBN mixup loss': loss.sum().item(),}
                
                loss.backward()
                self.optimizer.step()

                # calculate accuracy
                accuracy = l * (predictions.max(1)[1] == labels1).float() + (1 - l) * (predictions.max(1)[1] == labels2).float()
                accuracy = accuracy.sum() / accuracy.shape[0]

                # log information 
                iter_info_print.update({'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])})
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])

                first_batch = (epoch == 0) and (step == 0)
                if first_batch or self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.classifier.named_parameters())
                    utils.print_grad(self.model.named_parameters())

            # evaluation on validation set
            if self.eval:
                val_acc = self.testing.run_val(epoch)
            else:
                val_acc = 0.0

            # checkpoint
            self.checkpoint.save(self.model, self.classifier, epoch, self.logger, acc=val_acc)

            # update scheduler
            self.scheduler.step()

        # save best model path
        self.checkpoint.save_best_model_path(self.logger)