from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn import metrics


class Measurer(ABC):
    def __init__(self, tag):
        self.tag = tag

    @abstractmethod
    def cal(self, preds, targs, ctx):
        pass


class AcuuMeasurer(Measurer):
    def __init__(self, tag="accu"):
        super().__init__(tag)
        self.type = "scalar"

    def cal(self, preds_dis, targs, ctx):
        preds = torch.max(preds_dis, dim=1)[1]
        return preds.eq(targs).float().mean()


class CWAcuuMeasurer(Measurer):
    def __init__(self, tag="cw-accu"):
        super().__init__(tag)
        self.type = "scalar"

    def cal(self, preds_dis, targs, ctx):
        lab_num = len(targs.unique())
        confusion_matrix = torch.zeros(lab_num, lab_num)
        preds = torch.max(preds_dis, dim=1)[1]
        for t, p in zip(targs.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        per_cls_accu = confusion_matrix.diag() / confusion_matrix.sum(1)
        # print(per_cls_accu)
        return torch.mean(per_cls_accu)


class RocAucScoreMeasurer(Measurer):
    def __init__(self, tag="RocAucScore"):
        super().__init__(tag)
        self.type = "scalar"

    def cal(self, preds_dis, targs, ctx):
        max_probs = torch.max(preds_dis, dim=1)[0]
        max_probs = max_probs.cpu().numpy()
        max_probs = np.float32(max_probs)
        targs = targs.cpu().numpy()
        ras = metrics.roc_auc_score(targs, max_probs)
        return torch.tensor(ras)
