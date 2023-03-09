from copy import deepcopy
from curses import KEY_SAVE
from cv2 import log
import torch
from functools import partial
import contextlib
import pdb
from torch.nn.functional import kl_div


def sample_bool_inds(n_heads, p):
    bool_inds = (torch.rand(n_heads) < p).type(torch.uint8)
    if bool_inds.type(torch.int).sum() == 0:
        rnd_ind = (torch.rand(1) * n_heads).type(torch.long)
        bool_inds[rnd_ind] = 1
    return bool_inds


def relation_loss(logits, labels, reg_type, T):
    batch_size, n_cats, n_heads = logits.shape
    if n_heads < 2:
        return 0
    all_probs = torch.softmax(logits / T, dim=1)
    label_inds = torch.ones(batch_size, n_cats).cuda()
    label_inds[range(batch_size), labels] = 0

    # removing the gt prob
    probs = all_probs * label_inds.unsqueeze(-1).detach() #unsqueeze用来形成列向量
    # re-normalize such that probs sum to 1
    probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)

    if 'l2' in reg_type:
        dist_mat = probs.unsqueeze(-1) - probs.unsqueeze(-2)
        dist_mat = dist_mat ** 2
        den = batch_size * (n_heads - 1) * n_heads
        loss = dist_mat.sum() / den
    elif 'cos' in reg_type:
        probs = probs / torch.sqrt(((
            all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))    # l2 normed
        cov_mat = torch.einsum('ijk,ijl->ikl', probs, probs)
        pairwise_inds = 1 - torch.eye(n_heads).cuda()
        den = batch_size * (n_heads - 1) * n_heads
        loss = (cov_mat * pairwise_inds).sum() / den
    elif 'js' in reg_type:
        loss, count = 0.0, 0
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                p1, p2 = probs[:, :, i], probs[:, :, j]
                interm = (p1 * torch.log(p2 + 1e-7)
                          + p2 * torch.log(p1 + 1e-7)) / 2
                count += 1
                loss -= interm.mean(0).sum()
        loss = loss / count
    elif 'distil':
        loss, count = 0, 0
        summed_targets = all_probs.sum(-1)
        for i in range(n_heads):
            targets = (summed_targets - all_probs[:, :, i]) / (n_heads - 1)
            tlogits = logits[:, :, i] / T
            max_tlogits = tlogits.max(dim=-1, keepdim=True)[0].detach()
            dif_tlogits = tlogits - max_tlogits
            log_prob = tlogits - max_tlogits - torch.log(
                torch.exp(dif_tlogits).sum(-1, keepdim=True))

            loss = loss - (targets * log_prob).sum(-1).mean()
            count += 1
        loss = loss / count

    if 'neg' in reg_type:
        loss = -loss
    return loss


class Ensemble(object):
    def __init__(self, models,prob=1, dropout=0):
        self.models = models
        self.prob = prob
        self.num_heads=len(models)
        self.eval()

    def embed(self, x, avg=False):
        outs = []
        with self.maybe_grad():
            for i in range(self.num_heads):
                emb_out = self.models[i].embed(x)
                if isinstance(emb_out, list):
                    outs.extend(emb_out)
                else:
                    outs.append(emb_out)
            output = sum(outs) / len(outs) if avg else outs
        return output

    def forward(self, x, avg=False, softmax=False, T=1):
        outs = []
        with torch.no_grad():
            for i in range(self.num_heads):
                out = self.models[i].forward(x)
                if softmax:
                    out = torch.softmax(out / T, -1)#对最后一维softmax
                outs.append(out)
            output = sum(outs) / len(outs) if avg else outs
        return output

    def train(self):
        self._train = True
        self.maybe_grad = contextlib.suppress #选择性忽略特定异常
        for i in range(len(self.models)):
            self.models[i].train()

    def eval(self):
        self._train = False
        self.maybe_grad = torch.no_grad
        for i in range(len(self.models)):
            self.models[i].eval()

    def to_device(self):
        for i in range(len(self.models)):
            self.models[i].cuda()

    def parameters(self):
        for i in range(len(self.models)):
            for p in self.models[i].parameters():
                yield p


class Distilation(object):
    def __init__(self, ensemble,criterion, T,shared=True):
        self.ensemble = ensemble
        self.T = T
        self.criterion=criterion
        self.ensemble.eval()
        self.shared = shared #multi-head

    def get_loss(self, x, logits,KL=False,domain_labels=None,logit_type='inv+spe'):
        if self.shared:
            with torch.no_grad():
                x =[deepcopy(x) for _ in range(self.ensemble.branch_num)]
                out = torch.stack(self.ensemble(x))
                if logit_type == 'inv+spe':
                    out = torch.sum(out,1)
                elif logit_type == 'inv':
                    out = out[:,0]
                elif logit_type == 'spe':
                    out = out[:,1]
                source_domain_num = len(out)
                targets=[]
                for index in range(source_domain_num):
                    targets.append(torch.softmax(out[index]/self.T,-1))
            if domain_labels is not None:
                targets = [targets[domain_labels[i]][i] for i in range(len(targets[0]))]
                targets = torch.stack(targets).detach()

            else:
                targets = (sum(targets)/source_domain_num).detach()
            if KL:
                loss = kl_div(logits,targets)
            else:
                loss =  self.criterion(logits/self.T,targets)
            return loss
        else:
            with torch.no_grad():
                out = self.ensemble(x)
            targets = torch.softmax(out/self.T,-1).detach()
            if KL:
                loss = kl_div(logits,targets)
            else:
                loss =  self.criterion(logits/self.T,targets)
            return loss

