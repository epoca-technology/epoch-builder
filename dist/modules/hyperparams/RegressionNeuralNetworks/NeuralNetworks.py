from typing import TypedDict
from modules.hyperparams.RegressionNeuralNetworks.DNN import DNN, IDNN
from modules.hyperparams.RegressionNeuralNetworks.LSTM import LSTM, ILSTM
from modules.hyperparams.RegressionNeuralNetworks.CLSTM import CLSTM, ICLSTM


# Type
class INeuralNetworks(TypedDict):
    DNN: IDNN
    LSTM: ILSTM
    CLSTM: ICLSTM



# Expose the neural networks
REGRESSION_NEURAL_NETWORKS = {
    "DNN":  DNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM,
}