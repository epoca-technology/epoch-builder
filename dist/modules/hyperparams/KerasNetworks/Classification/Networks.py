from modules._types.hyperparams_types import IKerasHyperparamsNetworks
from modules.hyperparams.KerasNetworks.Classification.DNN import DNN
from modules.hyperparams.KerasNetworks.Classification.CNN import CNN
from modules.hyperparams.KerasNetworks.Classification.LSTM import LSTM
from modules.hyperparams.KerasNetworks.Classification.CLSTM import CLSTM




# Expose the neural networks
CLASSIFICATION_NETWORKS: IKerasHyperparamsNetworks = {
    "DNN":  DNN,
    "CNN":  CNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM
}