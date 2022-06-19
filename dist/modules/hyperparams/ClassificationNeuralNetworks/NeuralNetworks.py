from typing import TypedDict
from modules.hyperparams.ClassificationNeuralNetworks.DNN import DNN, IDNN
from modules.hyperparams.ClassificationNeuralNetworks.CNN import CNN, ICNN
from modules.hyperparams.ClassificationNeuralNetworks.LSTM import LSTM, ILSTM
from modules.hyperparams.ClassificationNeuralNetworks.CLSTM import CLSTM, ICLSTM


# Type
class INeuralNetworks(TypedDict):
    DNN: IDNN
    CNN: ICNN
    LSTM: ILSTM
    CLSTM: ICLSTM



# Expose the neural networks
CLASSIFICATION_NEURAL_NETWORKS = {
    "DNN":  DNN,
    "CNN":  CNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM,
}