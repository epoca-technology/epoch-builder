from typing import TypedDict, Union, List






# Keras Model Configuration
#
class IKerasModelConfig(TypedDict):
    # The name of the Keras Model. If it doesn't exist it will raise an error.
    name: str

    # Units
    units: Union[List[int], None]

    # Dropout rates
    dropout_rates: Union[List[float], None]

    # Number of predictions the model will output. This prediction is not set in the 
    # RegressionTraining.json file. However, it is populated once the RegressionTraining
    # instance is initialized.
    predictions: Union[int, None]



