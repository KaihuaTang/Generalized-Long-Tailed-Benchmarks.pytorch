######################################
#         Kaihua Tang
######################################

from test_baseline import test_baseline
from test_tde import test_tde
from test_la import test_la

def test_loader(config):
    if config['testing_opt']['type'] in ('baseline'):
        return test_baseline
    elif config['testing_opt']['type'] in ('TDE'):
        return test_tde
    elif config['testing_opt']['type'] in ('LA'):
        return test_la
    else:
        raise ValueError('Wrong Test Pipeline')

