import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class cosine_classifer(nn.Module):
    def __init__(self, indim=1, outdim=1, scale = 20, imprint_weight = None): 
        super(cosine_classifer, self).__init__()
        if imprint_weight is None:
            weight = torch.FloatTensor(outdim, indim).normal_(0.0, np.sqrt(2.0/indim)) 
        else:
            weight = imprint_weight 
        self.scale = scale
        self.weight=nn.Parameter(weight.data, requires_grad=True)
    def forward(self, x):
        x_norm = F.normalize(x,p=2, dim=1)
        weight_norm = F.normalize(self.weight,p=2,dim=1) 
        cos_sim = torch.mm(x_norm, weight_norm.t()) #t() :transposition
        return self.scale * cos_sim

