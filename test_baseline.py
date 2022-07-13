######################################
#         Kaihua Tang
######################################

import os

import torch
import torch.nn as nn
import torch.nn.functional as F 

import utils.general_utils as utils
from data.dataloader import get_loader

INVALID_ATTRIBUTE_DATASET = ('MSCOCO-LT', 'MSCOCO-BL')

class acc_by_splits():
    def __init__(self, logger, stype):
        self.logger = logger
        self.stype = stype
    
    def print_score(self, predictions, labels, num_class, split, csv_results=None, premask=False):
        split_set = sorted(list(set(split.tolist())))
        self.logger.info('------------- Detailed Splits by {} -----------'.format(self.stype))

        output_log = 'Recall/AC ==> '
        for s in split_set:
            s_mask = (split==s)
            if premask:
                s_acc = utils.calculate_recall(predictions[s_mask].view(-1), labels[s_mask].view(-1))  
            else:
                s_acc = utils.calculate_recall(predictions.view(-1), labels.view(-1), split_mask=s_mask)  
            output_log = output_log + '{}_{} Acc : {:7.4f} / {:5d}, '.format(self.stype, str(s), s_acc, s_mask.sum().item())
            if csv_results is not None:
                csv_results['recall'].append('{:.4f}'.format(s_acc))
        self.logger.info(output_log)

        output_log = 'Precision ==> '
        for s in split_set:
            s_mask = (split==s)
            if premask:
                s_prc = utils.calculate_precision(predictions[s_mask].view(-1), labels[s_mask].view(-1), num_class)
            else:
                s_prc = utils.calculate_precision(predictions.view(-1), labels.view(-1), num_class, split_mask=s_mask)
            output_log = output_log + '{}_{} Prc : {:7.4f} / {:5d}, '.format(self.stype, str(s), s_prc, s_mask.sum().item())
            if csv_results is not None:
                csv_results['precision'].append('{:.4f}'.format(s_prc))
        self.logger.info(output_log)

class test_baseline():
    def __init__(self, config, logger, model, classifier, val=False, specify_testset=None, add_ckpt=None):
        self.config = config
        self.logger = logger
        self.model = model
        self.classifier = classifier
        self.add_ckpt = add_ckpt

        self.save_all = self.config['saving_opt']['save_all']

        # get dataloader
        if val:
            self.phase = 'val'
            self.loader = get_loader(config, 'val', config['dataset']['testset'], logger)
        elif specify_testset is not None:
            self.phase = 'test'
            self.loader = get_loader(config, 'test', specify_testset, logger)
        else:
            self.phase = 'test'
            self.loader = get_loader(config, 'test', config['dataset']['testset'], logger)


    def run_test(self):
        currect_split = self.config['dataset']['testset']
        self.logger.info('------------- Start Testing at Split: {} -----------'.format(currect_split))

        if currect_split in ('test_bl', 'test_bbl'):
            frq_accs = acc_by_splits(self.logger, 'frequency')
        if currect_split in ('test_bbl'):
            if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                att_accs = acc_by_splits(self.logger, 'attribute')

        # set model to evaluation
        self.model.eval()
        self.classifier.eval()

        # run batch
        with torch.no_grad():
            all_preds = []
            all_labs = []
            all_frqs = []
            all_atts = []
            
            save_features = []
            save_preds = []
            save_labs = []
            save_frqs = []
            save_atts = []
            save_inds = []

            for _, (inputs, labels, freq_labels, attributes, indexes) in enumerate(self.loader):
                
                # additional inputs
                inputs, labels, freq_labels, attributes = inputs.cuda(), labels.cuda(), freq_labels.cuda(), attributes.cuda()
                add_inputs = {}
   
                features = self.model(inputs)
                predictions = self.classifier(features, add_inputs)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                all_preds.append(predictions)
                all_labs.append(labels)
                all_frqs.append(freq_labels)
                all_atts.append(attributes)

                # save output
                if self.save_all:
                    save_features.append(features.detach().clone().cpu())
                    save_preds.append(predictions.detach().clone().cpu())
                    save_labs.append(labels.detach().clone().cpu())
                    save_frqs.append(freq_labels.detach().clone().cpu())
                    save_atts.append(attributes.detach().clone().cpu())
                    save_inds.append(indexes.detach().clone().cpu())
            
            # save output
            if self.save_all:
                self.logger.info('============ Start Saving Test Outputs ===============')
                save_outputs = {'save_features': torch.cat(save_features, dim=0),
                                'save_preds' : torch.cat(save_preds, dim=0),
                                'save_labs' : torch.cat(save_labs, dim=0),
                                'save_frqs' : torch.cat(save_frqs, dim=0),
                                'save_atts' : torch.cat(save_atts, dim=0),
                                'save_inds' : torch.cat(save_inds, dim=0), }
                model_path = os.path.join(self.config['output_dir'], 'save_outputs_{}.pth'.format(self.config['dataset']['testset']))
                torch.save(save_outputs, model_path)
                self.logger.info('============ Test Outputs Saved to {} ==============='.format(model_path))


            all_preds = torch.cat(all_preds, dim=0)
            num_class = all_preds.shape[-1]
            all_preds = all_preds.max(-1)[1].view(-1)
            all_labs = torch.cat(all_labs, dim=0).view(-1)
            all_frqs = torch.cat(all_frqs, dim=0).view(-1)
            all_atts = torch.cat(all_atts, dim=0).view(-1)

            csv_results = {'recall': [currect_split, 'recall'], 
                            'precision': [currect_split, 'precision'], 
                            'f1': [currect_split, 'f1']}

            # frequency splits by intra-class attributes
            if currect_split in ('test_bl', 'test_bbl'):
                frq_accs.print_score(all_preds, all_labs, num_class, split=all_frqs, csv_results=csv_results)
            if currect_split in ('test_bbl'):
                if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                    att_accs.print_score(all_preds, all_labs, num_class, split=all_atts, csv_results=csv_results, premask=True)

            # overall performance
            total_num = all_preds.shape[0]
            # calculate recall
            recall_score = utils.calculate_recall(all_preds, all_labs)
            self.logger.info('Test Complete ==> Overall Recall/AC : {:9.4f}, Number Samples : {:9d}'.format(recall_score, total_num))
            csv_results['recall'].append('{:.4f}'.format(recall_score))
            # calculate precision
            precision_score = utils.calculate_precision(all_preds, all_labs, num_class)
            self.logger.info('Test Complete ==> Overall Precision : {:9.4f}, Number Samples : {:9d}'.format(precision_score, total_num))
            csv_results['precision'].append('{:.4f}'.format(precision_score))
            # F1 score
            F1_score = utils.calculate_f1(recall_score, precision_score)
            self.logger.info('Test Complete ==> Overall  F1 Score : {:9.4f}, Number Samples : {:9d}'.format(F1_score, total_num))
            csv_results['f1'].append('{:.4f}'.format(F1_score))

            # save csv results
            self.logger.write_results(csv_results['recall'])
            self.logger.write_results(csv_results['precision'])
            self.logger.write_results(csv_results['f1'])


        # set back to training mode again
        self.model.train()
        self.classifier.train()
        return recall_score
        
    def run_val(self, epoch):
        self.logger.info('------------- Start Validation at Epoch {} -----------'.format(epoch))
        frq_accs = acc_by_splits(self.logger, 'frequency')
        if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
            att_accs = acc_by_splits(self.logger, 'attribute')

        # set model to evaluation
        self.model.eval()
        self.classifier.eval()

        # run batch
        with torch.no_grad():
            all_preds = []
            all_labs = []
            all_frqs = []
            all_atts = []

            for _, (inputs, labels, freq_labels, attributes, indexes) in enumerate(self.loader):
                # additional inputs
                inputs, labels, freq_labels, attributes = inputs.cuda(), labels.cuda(), freq_labels.cuda(), attributes.cuda()
                add_inputs = {}
   
                features = self.model(inputs)
                predictions = self.classifier(features, add_inputs)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                all_preds.append(predictions)
                all_labs.append(labels)
                all_frqs.append(freq_labels)
                all_atts.append(attributes)

        
            all_preds = torch.cat(all_preds, dim=0)
            num_class = all_preds.shape[-1]
            all_preds = all_preds.max(-1)[1].view(-1)
            all_labs = torch.cat(all_labs, dim=0).view(-1)
            all_frqs = torch.cat(all_frqs, dim=0).view(-1)
            all_atts = torch.cat(all_atts, dim=0).view(-1)

            # overall performance
            total_num = all_preds.shape[0]
            # calculate recall
            recall_score = utils.calculate_recall(all_preds, all_labs)
            self.logger.info('Test Complete ==> Overall Recall/AC : {:9.4f}, Number Samples : {:9d}'.format(recall_score, total_num))
            # calculate precision
            precision_score = utils.calculate_precision(all_preds, all_labs, num_class)
            self.logger.info('Test Complete ==> Overall Precision : {:9.4f}, Number Samples : {:9d}'.format(precision_score, total_num))
            # F1 score
            F1_score = utils.calculate_f1(recall_score, precision_score)
            self.logger.info('Test Complete ==> Overall  F1 Score : {:9.4f}, Number Samples : {:9d}'.format(F1_score, total_num))

            # frequency splits by intra-class attributes
            frq_accs.print_score(all_preds, all_labs, num_class, split=all_frqs)
            if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                att_accs.print_score(all_preds, all_labs, num_class, split=all_atts, premask=True)

        # set back to training mode again
        self.model.train()
        self.classifier.train()
        return recall_score