######################################
#         Kaihua Tang
######################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class ClassifierRIDE(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, num_experts=3, use_norm=True):
        super(ClassifierRIDE, self).__init__()
        self.num_experts = num_experts
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(feat_dim, num_classes) for _ in range(num_experts)])
            s = 30
        else:
            self.linears = nn.ModuleList([nn.Linear(feat_dim, num_classes) for _ in range(num_experts)])
            s = 1
        self.s = s
    def forward(self, x, add_inputs=None, index=None):
        if index is None:
            logits = []
            for ind in range(self.num_experts):
                logit = (self.linears[ind])(x[:, ind, :])
                logits.append(logit * self.s)
            y = torch.stack(logits, dim=1).mean(dim=1)
            return y, logits
        else:
            logit = (self.linears[index])(x)
            return logit


def create_model(feat_dim=2048, num_classes=1000, num_experts=3, use_norm=True):
    model = ClassifierRIDE(feat_dim=feat_dim, num_classes=num_classes, 
                            num_experts=num_experts, use_norm=use_norm)
    return model