from typing import TypedDict
from modules.hyperparams.RegressionNeuralNetworks.DNN import DNN, IDNN
from modules.hyperparams.RegressionNeuralNetworks.CNN import CNN, ICNN
from modules.hyperparams.RegressionNeuralNetworks.LSTM import LSTM, ILSTM
from modules.hyperparams.RegressionNeuralNetworks.CLSTM import CLSTM, ICLSTM


# Type
class INeuralNetworks(TypedDict):
    DNN: IDNN
    CNN: ICNN
    LSTM: ILSTM
    CLSTM: ICLSTM



# Expose the neural networks
REGRESSION_NEURAL_NETWORKS = {
    "DNN":  DNN,
    "CNN":  CNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM,
}