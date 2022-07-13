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

# hard example mining

class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets, requires_weight = False, weight_base = 0):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight + weight_base
        if requires_weight:
            return loss, loss_weight
        return loss

class train_lff():
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
        model_b = utils.source_import(model_file).create_model(**model_args)
        model_d = utils.source_import(model_file).create_model(**model_args)
        classifier_b = utils.source_import(classifier_file).create_model(**classifier_args)
        classifier_d = utils.source_import(classifier_file).create_model(**classifier_args)

        model_b = nn.DataParallel(model_b).cuda()
        model_d = nn.DataParallel(model_d).cuda()
        classifier_b = nn.DataParallel(classifier_b).cuda()
        classifier_d = nn.DataParallel(classifier_d).cuda()

        # other initialization
        self.algorithm_opt = config['algorithm_opt']
        self.config = config
        self.logger = logger
        self.model_b = model_b
        self.model_d = model_d
        self.classifier_b = classifier_b
        self.classifier_d = classifier_d
        self.optimizer_b = create_optimizer(model_b, classifier_b, logger, config)
        self.optimizer_d = create_optimizer(model_d, classifier_d, logger, config)
        self.scheduler_b = create_scheduler(self.optimizer_b, logger, config)
        self.scheduler_d = create_scheduler(self.optimizer_d, logger, config)
        self.eval = eval
        self.training_opt = config['training_opt']

        self.checkpoint = Checkpoint(config)

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', config['dataset']['testset'], logger)

        # get loss
        self.loss_fc = nn.CrossEntropyLoss(reduction='none')
        # biased loss
        self.loss_bias = GeneralizedCELoss()
        

        # set eval
        if self.eval:
            test_func = test_loader(config)
            self.testing = test_func(config, logger, model_d, classifier_d, val=True)


    def run(self):
        # Start Training
        self.logger.info('=====> Start Baseline Training')

        # logit adjustment
        logit_adj = utils.compute_adjustment(self.train_loader, self.algorithm_opt['tro'])
        logit_adj.requires_grad = False

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))

            # preprocess for each epoch
            total_batch = len(self.train_loader)

            for step, (inputs, labels, _, _) in enumerate(self.train_loader):
                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                
                # additional inputs
                inputs, labels = inputs.cuda(), labels.cuda()
                add_inputs = {}
                batch_size = inputs.shape[0]
                add_inputs['logit_adj'] = logit_adj.to(inputs.device).view(1, -1).repeat(batch_size, 1)

                # biased prediction
                predictions_b = self.classifier_b(self.model_b(inputs), add_inputs)
                # targeted prediction
                predictions_d = self.classifier_d(self.model_d(inputs), add_inputs)
                
                # calculate hard exampling mining weight
                loss_b = self.loss_fc(predictions_b, labels).detach()
                loss_d = self.loss_fc(predictions_d, labels).detach()
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                
                # calculate loss
                # biased model 
                loss_b_update = self.loss_bias(predictions_b, labels)
                loss_d_update = self.loss_fc(predictions_d, labels) * loss_weight.cuda().detach()
                loss = loss_b_update.mean() + loss_d_update.mean()
                
                iter_info_print = {'biased loss' : loss_b_update.mean().item(), 'target loss': loss_d_update.mean().item()}
                
                loss.backward()
                self.optimizer_b.step()
                self.optimizer_d.step()

                # calculate accuracy
                accuracy = (predictions_d.max(1)[1] == labels).sum().float() / predictions_d.shape[0]

                # log information 
                iter_info_print.update({'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer_d.param_groups[0]['lr'])})
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])

            # evaluation on validation set
            if self.eval:
                val_acc = self.testing.run_val(epoch)
            else:
                val_acc = 0.0

            # checkpoint
            self.checkpoint.save(self.model_d, self.classifier_d, epoch, self.logger, acc=val_acc)

            # update scheduler
            self.scheduler_b.step()
            self.scheduler_d.step()

        # save best model path
        self.checkpoint.save_best_model_path(self.logger)