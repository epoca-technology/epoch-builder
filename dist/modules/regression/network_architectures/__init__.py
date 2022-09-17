from modules._types import IRegressionTrainingConfigNetworksByCategory
from modules.regression.network_architectures.DNN import DNN
from modules.regression.network_architectures.CDNN import CDNN
from modules.regression.network_architectures.LSTM import LSTM
from modules.regression.network_architectures.CLSTM import CLSTM



# Expose the neural networks
NETWORKS_BY_CATEGORY: IRegressionTrainingConfigNetworksByCategory = {
    "DNN":  DNN,
    "CDNN":  CDNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM
}