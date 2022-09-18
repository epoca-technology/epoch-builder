from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.regression.RegressionTrainingConfig import RegressionTrainingConfig



# GENERATE REGRESSION TRAINING CONFIGS
# Args:
#   None
endpoint_name: str = "GENERATE REGRESSION TRAINING CONFIGS"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)




# Initialize the Epoch
Epoch.init()



# Generate the configurations
RegressionTrainingConfig.generate()




# End of Script
Utils.endpoint_footer(endpoint_name)