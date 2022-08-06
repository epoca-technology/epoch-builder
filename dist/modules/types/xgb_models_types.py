from typing import Literal, TypedDict, Union, List, Tuple







## Configuration ##




# XGBoost Model Configuration
# The configuration that will be used to build the XGB Model.
class IXGBModelConfig(TypedDict):
    # The name of the XGB Model. If it doesn't exist it will raise an error.
    name: str

    # 
    # @TODO

    # Regression Model Type
    # Default: will generate all predictions in one go.
    # Autoregressive: will generate 1 prediction at a time and feed it to itself as an input 
    autoregressive: Union[bool, None]

    # Lookback used as the model's input. This lookback is not set in the 
    # RegressionTraining.json file. However, it is populated once the RegressionTraining
    # instance is initialized.
    # Also keep in mind that this property only exists Regressions.
    lookback: Union[int, None]

    # Only used for not autoregressive regressions
    # Number of predictions the model will output. This prediction is not set in the 
    # RegressionTraining.json file. However, it is populated once the RegressionTraining
    # instance is initialized.
    # Also keep in mind that this property only exists Regressions.
    predictions: Union[int, None]

    # Number of features used for the input layer of a Classification Network. This value is 
    # not set in the ClassificationTraining.json file. Instead, it is populated once the 
    # ClassificationTraining instance is initialized.
    features_num: Union[int, None]







## Training Configuration ##



# XGB Model Training Type Configuration
# Based on the type of training (hyperparams|shortlist), different training settings will be used.
# For more information regarding these args, view the KerasTraining.ipynb notebook.
class IXGBTrainingTypeConfig(TypedDict):
    # The split that will be applied to the dataset that will generate the train and test datasets
    train_split: float

    # 
    # @TODO










## Training History ##



# Training History
# The dictionary built once the training is completed. The properties adapt accordingly 
# based on the loss and metric functions used.
class IXGBModelTrainingHistory(TypedDict):
    # @TODO
    something: List[float]







## Model Summary ##






# Model Class Name
IXGBModelClassName = Literal["Sequential"]



# Model Summary
# Extracts all the relevant information from a trained model.
class IXGBModelSummary(TypedDict):
    # The name of the class used to create the model.
    model_class: IXGBModelClassName

    # @TODO










