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

class train_irm_dual():
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
        self.algorithm_opt = config['algorithm_opt']
        self.args = args
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


    def run(self):
        # Start Training
        self.logger.info('=====> Start IRM Loss with Dual Env Training')

        # preprocess for each epoch
        env1_loader, env2_loader = self.train_loader
        assert len(env1_loader) == len(env2_loader)
        total_batch = len(env1_loader)
        total_image = len(env1_loader.dataset)

        # run epoch
        num_epoch = self.training_opt['num_epochs']
        for epoch in range(num_epoch):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            self.logger.info('--------------- Environment Type {} -----------'.format(self.algorithm_opt['env_type']))
            # saving training info for environments building
            all_ind = []
            all_lab = []
            all_prb = []
            all_lgt = []

            for step, ((inputs1, labels1, _, indexs1), (inputs2, labels2, _, indexs2)) in enumerate(zip(env1_loader, env2_loader)):
                iter_info_print = {}

                self.optimizer.zero_grad()

                # additional inputs
                inputs = torch.cat([inputs1, inputs2], dim=0).cuda()
                labels = torch.cat([labels1, labels2], dim=0).cuda()
                indexs = torch.cat([indexs1, indexs2], dim=0).cuda()
                add_inputs = {}

                features = self.model(inputs)
                predictions = self.classifier(features, add_inputs)
                
                # calculate loss
                loss_ce = self.loss_fc(predictions, labels)
                iter_info_print[self.training_opt['loss']] = loss_ce.sum().item()

                # irm loss
                scale = torch.ones((1, predictions.size(-1))).cuda().requires_grad_()
                loss_tmp = self.loss_fc(predictions * scale, labels)
                grad = torch.autograd.grad(loss_tmp, [scale], create_graph=True)[0]
                loss_irm = grad.abs().sum() * self.algorithm_opt['irm_weight']
                # loss_irm = grad.pow(2).mean().sqrt()
                iter_info_print['IRM'] = loss_irm.sum().item()

                # backward
                loss = loss_ce + loss_irm
                loss.backward()
                self.optimizer.step()

                # calculate accuracy
                accuracy = (predictions.max(1)[1] == labels).sum().float() / predictions.shape[0]

                # save info for environment spliting
                all_lgt.append(predictions.detach().clone().cpu())
                predictions = predictions.softmax(-1)
                gt_score = torch.gather(predictions, 1, torch.unsqueeze(labels, 1)).view(-1)
                all_ind.append(indexs.detach().clone().cpu())
                all_lab.append(labels.detach().clone().cpu())
                all_prb.append(gt_score.detach().clone().cpu())

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

            # update irm weight
            if ('decay_milestones' in self.algorithm_opt) and (epoch in self.algorithm_opt['decay_milestones']):
                self.algorithm_opt['irm_weight'] = self.algorithm_opt['irm_weight'] * self.algorithm_opt['irm_alpha']
                self.logger.info('IRM weight is changed to {}'.format(self.algorithm_opt['irm_weight']))

            # save env score
            env_score_memo = {}

            if self.algorithm_opt['always_update'] or (epoch in self.algorithm_opt['update_milestones']):
                # update env mask
                self.all_ind = torch.cat(all_ind, dim=0)
                self.all_lab = torch.cat(all_lab, dim=0)
                self.all_prb = torch.cat(all_prb, dim=0)
                self.all_lgt = torch.cat(all_lgt, dim=0)

                # save env_score
                env_score_memo['label_{}'.format(epoch)] = self.all_lab.tolist()
                env_score_memo['prob_{}'.format(epoch)] = self.all_prb.tolist()
                env_score_memo['idx_{}'.format(epoch)] = self.all_ind.tolist()

                if self.algorithm_opt['env_type'] == 'correctness':
                    self.update_env_by_correct(env1_loader, env2_loader, total_image)
                elif self.algorithm_opt['env_type'] in ('inter', 'intra', 'inter_intra'):
                    self.update_env_by_score(env1_loader, env2_loader, total_image)            
                elif self.algorithm_opt['env_type'] == 'simple_aug':
                    self.update_env_by_uniform(env1_loader, env2_loader, total_image)
                else:
                    raise ValueError('Wrong Env Type')

            # checkpoint
            self.checkpoint.save(self.model, self.classifier, epoch, self.logger, acc=val_acc, add_dict=env_score_memo)

            # update scheduler
            self.scheduler.step()

        # save best model path
        self.checkpoint.save_best_model_path(self.logger)



    def update_env_by_uniform(self, env1_loader, env2_loader, total_image):
        # simple sample dataset 2 times at each epoch
        all_scores = torch.zeros(total_image).fill_(1.0)
        self.logger.info('Env1 and Env2 have size {} and {}'.format(all_scores.sum().item(), all_scores.sum().item()))
        env1_loader.sampler.set_parameter(all_scores)
        env2_loader.sampler.set_parameter(all_scores)


    def update_env_by_correct(self, env1_loader, env2_loader, total_image):
        # seperate environments by correct/wrong prediction
        correct_index = self.all_ind[self.all_lgt.max(1)[1] == self.all_lab].cpu().tolist()
        env1_score = torch.zeros(total_image)
        for idx in correct_index:
            env1_score[idx] = 1.0
        env2_score = 1.0 - env1_score
        self.logger.info('Env1 and Env2 have size {} and {}'.format(env1_score.sum().item(), env2_score.sum().item()))
        env1_loader.sampler.set_parameter(env1_score)
        env2_loader.sampler.set_parameter(env2_score)


    def update_env_by_score(self, env1_loader, env2_loader, total_image):
        # seperate environments by inter-score + intra-score
        all_ind, all_lab, all_prb = self.all_ind.tolist(), self.all_lab.tolist(), self.all_prb.tolist()
        all_cat = list(set(all_lab))
        all_cat.sort()
        cat_socres = {cat:{} for cat in all_cat}
        all_scores = {}
        for ind, lab, prb in zip(all_ind, all_lab, all_prb):
            cat_socres[lab][ind] = prb
            all_scores[ind] = prb

        
        # baseline distribution
        env1_score = torch.zeros(total_image).fill_(1.0)
        env2_score = torch.zeros(total_image).fill_(1.0)
        # inverse distribution
        if self.algorithm_opt['env_type'] in ('inter', 'inter_intra'):
            inter_weight = self.generate_inter_weight(all_scores, total_image, tg_scale=self.algorithm_opt['sample_scale'])
            env2_score = env2_score * inter_weight
        if self.algorithm_opt['env_type'] in ('intra', 'inter_intra'):
            intra_weight = self.generate_intra_weight(cat_socres, total_image, tg_scale=self.algorithm_opt['sample_scale'])
            env2_score = env2_score * intra_weight

        env1_loader.sampler.set_parameter(env1_score)
        env2_loader.sampler.set_parameter(env2_score)


    def generate_inter_weight(self, all_scores, total_image, tg_scale=4.0):
        # normalize
        inter_weight = torch.zeros(total_image).fill_(1.0)
        for ind, prb in all_scores.items():
            inter_weight[ind] = prb
        inter_weight = inter_weight - inter_weight.min()
        inter_weight = inter_weight / (inter_weight.max() + 1e-9)

        # use Pareto principle to determine the scale parameter
        inter_weight = (1.0 - inter_weight).abs() + 1e-5
        head_mean = torch.topk(inter_weight, k=int(total_image * 0.8), largest=False)[0].mean().item()
        tail_mean = torch.topk(inter_weight, k=int(total_image * 0.2), largest=True )[0].mean().item()
        scale = tail_mean / head_mean + 1e-5
        exp_scale = torch.FloatTensor([tg_scale]).log() / torch.FloatTensor([scale]).log()
        exp_scale = exp_scale.clamp(min=1, max=10)
        self.logger.info('Inter Score Original Head (80) Tail (20) Scale is {}'.format(scale))
        self.logger.info('Inter Score Target   Head (80) Tail (20) Scale is {}'.format(tg_scale))
        self.logger.info('Inter Score Exp Scale is {}'.format(exp_scale.item()))
        inter_weight = inter_weight ** exp_scale
        inter_weight = inter_weight + 1e-12
        inter_weight = inter_weight / inter_weight.sum()
        return inter_weight


    def generate_intra_weight(self, cat_socres, total_image, tg_scale=4.0):
        # normalize
        intra_weight = torch.zeros(total_image).fill_(0.0)
        for cat, cat_items in cat_socres.items():
            cat_size = len(cat_items)
            if cat_size < 5:
                for ind in list(cat_items.keys()):
                    intra_weight[ind] = 1.0 / max(cat_size, 1.0)
                continue
            cat_inds = list(cat_items.keys())
            cat_scores = torch.FloatTensor([cat_items[ind] for ind in cat_inds])
            cat_scores = cat_scores - cat_scores.min()
            cat_scores = cat_scores / (cat_scores.max() + 1e-9)

            # use Pareto principle to determine the scale parameter
            cat_scores = (1.0 - cat_scores).abs() + 1e-5
            head_mean = torch.topk(cat_scores, k=int(cat_size * 0.8), largest=False)[0].mean().item()
            tail_mean = torch.topk(cat_scores, k=int(cat_size * 0.2), largest=True )[0].mean().item()
            scale = tail_mean / head_mean + 1e-5
            exp_scale = torch.FloatTensor([tg_scale]).log() / torch.FloatTensor([scale]).log()
            exp_scale = exp_scale.clamp(min=1, max=10)
            if int(cat) == 0:
                self.logger.info('Intra Score at Cat-{} Original Head (80) Tail (20) Scale is {}'.format(cat, scale))
                self.logger.info('Intra Score at Cat-{} Target   Head (80) Tail (20) Scale is {}'.format(cat, tg_scale))
                self.logger.info('Intra Score at Cat-{} Exp Scale is {}'.format(cat, exp_scale.item()))
            cat_scores = cat_scores ** exp_scale
            cat_scores = cat_scores + 1e-12
            cat_scores = cat_scores / cat_scores.sum()
            for ind, score in zip(cat_inds, cat_scores.tolist()):
                intra_weight[ind] = score
        self.logger.info('Intra Total Score {}, which should be equal to NUM_CLASS'.format(intra_weight.sum().item()))
        return intra_weight
        


    

