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

class train_stage2():
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
        self.optimizer = create_optimizer_stage2(model, classifier, logger, config)
        self.scheduler = create_scheduler(self.optimizer, logger, config)
        self.eval = eval
        self.training_opt = config['training_opt']

        self.checkpoint = Checkpoint(config)

        if config['classifiers']['type'] == 'LWS':
            self.checkpoint.load(self.model, self.classifier, args.load_dir, logger)
        else:
            self.checkpoint.load_backbone(self.model, args.load_dir, logger)

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', config['dataset']['testset'], logger)

        # get loss
        self.loss_fc = create_loss(logger, config, self.train_loader)

        # set eval
        if self.eval:
            test_func = test_loader(config)
            self.testing = test_func(config, logger, model, classifier, val=True)


    def run(self):
        # Start Training
        self.logger.info('=====> Start Stage 2 Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))

            # preprocess for each epoch
            total_batch = len(self.train_loader)

            for step, (inputs, labels, _, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # additional inputs
                inputs, labels = inputs.cuda(), labels.cuda()
                add_inputs = {}

                features = self.model(inputs)
                predictions = self.classifier(features, add_inputs)
                
                # calculate loss
                loss = self.loss_fc(predictions, labels)
                iter_info_print = {self.training_opt['loss'] : loss.sum().item(),}
                
                loss.backward()
                self.optimizer.step()

                # calculate accuracy
                accuracy = (predictions.max(1)[1] == labels).sum().float() / predictions.shape[0]

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