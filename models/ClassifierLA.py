######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models

import math

class ClassifierLA(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, posthoc=False, loss=False):
        super(ClassifierLA, self).__init__()

        self.posthoc = posthoc
        self.loss = loss
        assert (self.posthoc and self.loss) == False
        assert (self.posthoc or self.loss) == True

        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x, add_inputs=None):
        y = self.fc(x)
        if self.training and self.loss:
            logit_adj = add_inputs['logit_adj']
            y = y + logit_adj
        if (not self.training) and self.posthoc:
            logit_adj = add_inputs['logit_adj']
            y = y - logit_adj
        return y


def create_model(feat_dim=2048, num_classes=1000, posthoc=True, loss=False):
    model = ClassifierLA(feat_dim=feat_dim, num_classes=num_classes, posthoc=posthoc, loss=loss)
    return model