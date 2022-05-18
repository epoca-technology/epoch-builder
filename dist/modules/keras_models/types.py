from typing import TypedDict, Union, List, Any, Tuple



# Keras Path
# A dictionary containing the path to all directories that will be used by Keras.
class IKerasPath(TypedDict):
    # Keras Assets
    # The root path for the assets
    assets: str

    # Keras Models
    # The path in which all regression and classification models are stored
    models: str

    # Classification Training Data
    # The path containing all the classification training data files.
    classification_training_data: str

    # Batched Training Certificates
    # Even though individual certificates are stored within the model's directory,
    # a batch is also saved on a different directory so multiple configurations can
    # be evaluated simultaneously.
    batched_training_certificates: str

    # Model Configurations
    # Even though this path is not used by the system yet, it is recommended to keep all
    # the relevant configuration files in this directory.
    model_configs: str







## Configuration ##


# Keras Model Configuration
# The configuration that will be used to build the Keras Model.
class IKerasModelConfig(TypedDict):
    # The name of the Keras Model. If it doesn't exist it will raise an error.
    name: str

    # Units
    units: Union[List[int], None]

    # Dropout rates
    dropout_rates: Union[List[float], None]

    # Activations
    activations: Union[List[str], None]

    # Lookback used as the model's input. This lookback is not set in the 
    # RegressionTraining.json file. However, it is populated once the RegressionTraining
    # instance is initialized.
    lookback: Union[int, None]

    # Number of predictions the model will output. This prediction is not set in the 
    # RegressionTraining.json file. However, it is populated once the RegressionTraining
    # instance is initialized.
    predictions: Union[int, None]







## Training History ##



# Training History
# The dictionary built once the training is completed. The properties adapt accordingly 
# based on the loss and metric functions used.
class IKerasModelTrainingHistory(TypedDict):
    loss: List[float]
    val_loss: List[float]

    # Regression Values
    mean_absolute_error: Union[List[float], None]
    val_mean_absolute_error: Union[List[float], None]
    mean_squared_error: Union[List[float], None]
    val_mean_squared_error: Union[List[float], None]

    # Classification Values
    categorical_accuracy: Union[List[float], None]
    val_categorical_accuracy: Union[List[float], None]







## Model Summary ##



# Model Optimizer Config
# The optimizer configuration used when the model was compiled. Keep in mind that
# all these values are stringified to ensure compatibility with the JSON file format.
class IKerasModelOptimizerConfig(TypedDict):
    name: str
    learning_rate: str
    decay: Union[str, None]
    beta_1: Union[str, None]
    beta_2: Union[str, None]
    epsilon: Union[str, None]
    amsgrad: Union[str, None]





# Model Loss Config
# The loss configuration used when the model was compiled. Keep in mind that
# all these values are stringified to ensure compatibility with the JSON file format.
class IKerasModelLossConfig(TypedDict):
    name: str
    reduction: Union[str, None]




# Model Layer
# A layer stacked with other layers within the model.
class IKerasModelLayer(TypedDict):
    name: str
    params: int
    input_shape: Tuple[int]
    output_shape: Tuple[int]
    trainable: bool





# Model Summary
# Extracts all the relevant information from a trained model.
class IKerasModelSummary(TypedDict):
    # The name of the class used to create the model.
    model_class: str

    # Optimizer Config
    optimizer_config: IKerasModelOptimizerConfig

    # Loss Config
    loss_config: IKerasModelLossConfig

    # Metrics
    metrics: List[str]

    # Input and output shapes
    input_shape: Tuple[int]
    output_shape: Tuple[int]

    # Layers
    layers: List[IKerasModelLayer]

    # Params
    total_params: int
    trainable_params: int
    non_trainable_params: int


