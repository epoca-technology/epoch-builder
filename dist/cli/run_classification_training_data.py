from modules.types import ITrainingDataConfig
from modules.epoch.Epoch import Epoch
from modules.classification_training_data.ClassificationTrainingData import ClassificationTrainingData


# Initialize the Epoch
Epoch.init()



# TRAINING DATA CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config: ITrainingDataConfig = Epoch.FILE.get_classification_training_data_config()



# TRAINING DATA INSTANCE
# The Instance of the Training Data that will be executed
training_data: ClassificationTrainingData = ClassificationTrainingData(config)



# TRAINING DATA EXECUTION
# Runs the Training Data for all models simultaneously. 
# Results as saved when the execution has completed. If the execution is interrupted before
# completion, results will not be saved.
print("TRAINING DATA RUNNING")
print(f"\n{training_data.id}:\n")
training_data.run()
print("\nTRAINING DATA COMPLETED")
