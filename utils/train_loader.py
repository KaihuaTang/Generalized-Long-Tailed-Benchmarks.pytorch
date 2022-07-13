######################################
#         Kaihua Tang
######################################

import imp
from train_baseline import train_baseline
from train_la import train_la
from train_bbn import train_bbn
from train_tde import train_tde
from train_ride import train_ride
from train_tade import train_tade
from train_ldam import train_ldam
from train_mixup import train_mixup

from train_lff import train_lff

from train_stage1 import train_stage1
from train_stage2 import train_stage2
from train_stage2_ride import train_stage2_ride

from train_irm_dual import train_irm_dual
from train_center_dual import train_center_dual
from train_center_single import train_center_single
from train_center_triple import train_center_triple
from train_center_tade import train_center_tade
from train_center_ride import train_center_ride
from train_center_ride_mixup import train_center_ride_mixup
from train_center_ldam_dual import train_center_ldam_dual
from train_center_dual_mixup import train_center_dual_mixup


def train_loader(config):
    if config['training_opt']['type'] in ('baseline', 'Focal'):
        return train_baseline
    elif config['training_opt']['type'] in ('LFF', 'LFFLA'):
        return train_lff
    elif config['training_opt']['type'] in ('LA', 'FocalLA'):
        return train_la
    elif config['training_opt']['type'] in ('BBN'):
        return train_bbn
    elif config['training_opt']['type'] in ('TDE'):
        return train_tde
    elif config['training_opt']['type'] in ('mixup'):
        return train_mixup
    elif config['training_opt']['type'] in ('LDAM'):
        return train_ldam
    elif config['training_opt']['type'] in ('RIDE'):
        return train_ride
    elif config['training_opt']['type'] in ('TADE'):
        return train_tade
    elif config['training_opt']['type'] in ('stage1'):
        return train_stage1
    elif config['training_opt']['type'] in ('crt_stage2', 'lws_stage2'):
        return train_stage2
    elif config['training_opt']['type'] in ('ride_stage2'):
        return train_stage2_ride
    elif config['training_opt']['type'] in ('center_dual', 'env_dual'):
        return train_center_dual
    elif config['training_opt']['type'] in ('center_single'):
        return train_center_single
    elif config['training_opt']['type'] in ('center_triple'):
        return train_center_triple
    elif config['training_opt']['type'] in ('center_LDAM_dual'):
        return train_center_ldam_dual
    elif config['training_opt']['type'] in ('center_dual_mixup'):
        return train_center_dual_mixup
    elif config['training_opt']['type'] in ('center_tade'):
        return train_center_tade
    elif config['training_opt']['type'] in ('center_ride'):
        return train_center_ride
    elif config['training_opt']['type'] in ('center_ride_mixup'):
        return train_center_ride_mixup
    elif config['training_opt']['type'] in ('irm_dual'):
        return train_irm_dual
    else:
        raise ValueError('Wrong Train Type')