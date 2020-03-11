from numpy import mean

class NetoworksLoss:
    def __init__(self):
        pass


    def mse_loss(y_true, y_pred):
        """
        Mean squared error (MSE) loss between the real (y_true) and predicted (y_pred) output
        - y_pred and y_real have the same size
        """
        return ((y_true - y_pred) ** 2).mean()