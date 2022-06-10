import torch
import torch.nn.functional as t_func


class Accuracy():
    def __init__(self):
        pass

    def compute(self, pred, gt):
        if len(list(pred.size())) == 2:
            pred = t_func.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
        corrects = (pred == gt).float().sum()
        total = float(gt.shape[0])
        score = corrects/total
        return score
