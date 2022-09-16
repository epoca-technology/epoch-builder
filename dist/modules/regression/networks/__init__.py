from modules._types import IRegressionTrainingConfigNetworksByCategory
from modules.regression.networks.DNN import DNN
from modules.regression.networks.CDNN import CDNN
from modules.regression.networks.LSTM import LSTM
from modules.regression.networks.CLSTM import CLSTM



# Expose the neural networks
NETWORKS_BY_CATEGORY: IRegressionTrainingConfigNetworksByCategory = {
    "DNN":  DNN,
    "CDNN":  CDNN,
    "LSTM":  LSTM,
    "CLSTM":  CLSTM
}