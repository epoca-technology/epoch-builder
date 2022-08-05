from modules.types.hyperparams_types import IKerasHyperparamsNetworks
from modules.hyperparams.KerasNetworks.Regression.DNN import DNN
from modules.hyperparams.KerasNetworks.Regression.CNN import CNN
from modules.hyperparams.KerasNetworks.Regression.LSTM import LSTM
from modules.hyperparams.KerasNetworks.Regression.CLSTM import CLSTM



# Expose the neural networks
REGRESSION_NETWORKS: IKerasHyperparamsNetworks = {
    "DNN":  DNN,
    "CNN":  CNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM
}