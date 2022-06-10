from sklearn.metrics import mean_absolute_error as mae

class MAE():
    def __init__(self):
        pass

    def compute(self, preds, labels):
        preds = preds.detach().numpy()
        labels = labels.detach().numpy()
        return mae(labels, preds)