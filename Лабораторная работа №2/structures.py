class Dataset:

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.features_amount = len(features[0])
        self.data_amount = len(features)


class Model:
    def __init__(self, dataset, weights, regularisation, initial_grad, diff_weights):
        self.features = dataset.features
        self.targets = dataset.targets
        self.weights = weights
        self.regularisation = regularisation
        self.features_amount = dataset.features_amount
        self.data_amount = len(dataset.features)
        self.initial_grad = initial_grad
        self.diff_weights = diff_weights
