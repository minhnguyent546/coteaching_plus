from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal

# Loss functions
def loss_coteaching(logits1, logits2, labels, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(logits1, labels, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(logits2, labels, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]])/float(num_remember)

    loss_1_update = F.cross_entropy(logits1[ind_2_update], labels[ind_2_update])
    loss_2_update = F.cross_entropy(logits2[ind_1_update], labels[ind_1_update])

    return torch.sum(loss_1_update), torch.sum(loss_2_update), pure_ratio_1, pure_ratio_2

def loss_coteaching_plus(logits1, logits2, labels, forget_rate, ind, noise_or_not, step):
    pred1 = torch.argmax(logits1, dim=1)
    pred2 = torch.argmax(logits2, dim=1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_indices = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_indices.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = ind * logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_indices)
    except:
        disagree_indices = disagree_indices[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_indices) > 0:
        updated_labels = labels[disagree_indices]
        updated_logits1 = logits1[disagree_indices]
        updated_logits2 = logits2[disagree_indices]

        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(updated_logits1, updated_logits2, updated_labels, forget_rate, ind_disagree, noise_or_not)
    else:
        updated_labels = labels
        updated_logits1 = logits1
        updated_logits2 = logits2

        cross_entropy_1 = F.cross_entropy(updated_logits1, updated_labels)
        cross_entropy_2 = F.cross_entropy(updated_logits2, updated_labels)

        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.shape[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.shape[0]

        pure_ratio_1 = np.sum(noise_or_not[ind]) / ind.shape[0]
        pure_ratio_2 = np.sum(noise_or_not[ind]) / ind.shape[0]

    return loss_1, loss_2, pure_ratio_1, pure_ratio_2

