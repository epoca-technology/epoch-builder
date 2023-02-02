from modules._types import IRegressionTrainingConfigNetworksByCategory
from modules.regression.network_architectures.CNN import CNN
from modules.regression.network_architectures.DNN import DNN
from modules.regression.network_architectures.CDNN import CDNN
from modules.regression.network_architectures.LSTM import LSTM
from modules.regression.network_architectures.BDLSTM import BDLSTM
from modules.regression.network_architectures.CLSTM import CLSTM
from modules.regression.network_architectures.GRU import GRU



# Expose the neural networks
NETWORKS_BY_CATEGORY: IRegressionTrainingConfigNetworksByCategory = {
    "CNN":  CNN,
    "DNN":  DNN,
    "CDNN":  CDNN,
    "LSTM":  LSTM,
    "BDLSTM":  BDLSTM,
    "CLSTM":  CLSTM,
    "GRU":  GRU
}