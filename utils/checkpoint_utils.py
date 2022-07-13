######################################
#         Kaihua Tang
######################################

import os
import json
import torch
import numpy as np 

class Checkpoint():
    def __init__(self, config):
        self.config = config
        self.best_epoch = -1
        self.best_performance = -1
        self.best_model_path = None 

    
    def save(self, model, classifier, epoch, logger, add_dict={}, acc=None):
        # only save at certain steps or the last epoch
        if epoch % self.config['checkpoint_opt']['checkpoint_step'] != 0 and epoch < (self.config['training_opt']['num_epochs'] - 1):
            return

        output = {
            'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'epoch': epoch,
        }
        output.update(add_dict)

        model_name = 'epoch_{}_'.format(epoch) + self.config['checkpoint_opt']['checkpoint_name']
        model_path = os.path.join(self.config['output_dir'], model_name)

        logger.info('Model at epoch {} is saved to {}'.format(epoch, model_path))
        torch.save(output, model_path)

        # update best model
        if acc is not None:
            if float(acc) > self.best_performance:
                self.best_epoch = epoch
                self.best_performance = float(acc)
                self.best_model_path = model_path
        else:
            # if acc is None, the newest is always the best
            self.best_epoch = epoch
            self.best_model_path = model_path
        
        logger.info('Best model is at epoch {} with accuracy {:9.3f}'.format(self.best_epoch, self.best_performance))


    def save_best_model_path(self, logger):
        logger.info('Best model is at epoch {} with accuracy {:9.3f} (Path: {})'.format(self.best_epoch, self.best_performance, self.best_model_path))
        with open(os.path.join(self.config['output_dir'], 'best_checkpoint'), 'w+') as f:
            f.write(self.best_model_path + ' ' + str(self.best_epoch) + ' ' + str(self.best_performance) + '\n')

    def load(self, model, classifier, path, logger):        
        if path.split('.')[-1] != 'pth':
            with open(os.path.join(path, 'best_checkpoint')) as f:
                path = f[0].split(' ')[0]

        checkpoint = torch.load(path, map_location='cpu')
        logger.info('Loading checkpoint pretrained with epoch {}.'.format(checkpoint['epoch']))

        model_state = checkpoint['model']
        classifier_state = checkpoint['classifier']

        model = self.load_module(model, model_state, logger)
        classifier = self.load_module(classifier, classifier_state, logger)
        return checkpoint


    def load_backbone(self, model, path, logger):      
        if path.split('.')[-1] != 'pth':
            with open(os.path.join(path, 'best_checkpoint')) as f:
                path = f[0].split(' ')[0]

        checkpoint = torch.load(path, map_location='cpu')
        logger.info('Loading checkpoint pretrained with epoch {}.'.format(checkpoint['epoch']))

        model_state = checkpoint['model']

        model = self.load_module(model, model_state, logger)
        return checkpoint

    
    def load_module(self, module, module_state, logger):
        x = module.state_dict()
        for key, _ in x.items():
            if key in module_state:
                x[key] = module_state[key]
                logger.info('Load {:>50} from checkpoint.'.format(key))
            elif 'module.' + key in module_state:
                x[key] = module_state['module.' + key]
                logger.info('Load {:>50} from checkpoint (rematch with module.).'.format(key))
            else:
                logger.info('WARNING: Key {} is missing in the checkpoint.'.format(key))

        module.load_state_dict(x)
        return module