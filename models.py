
from sklearn.neural_network import MLPRegressor, MLPClassifier


def nn_model():
    parameters = {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (20,2), 'learning_rate_init': 0.1, 'solver': 'sgd'}
    mlp = MLPClassifier(** parameters)
    return mlp
