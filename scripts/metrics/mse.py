from sklearn.metrics import mean_squared_error as mse

class MSE():
    def __init__(self):
        pass

    def compute(self, preds, labels):
        preds = preds.detach().numpy()
        labels = labels.detach().numpy()
        return mse(labels, preds)