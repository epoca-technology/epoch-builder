from typing import TypedDict, Union, List, Tuple



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

    # Regression Training Configurations
    # Even though this path is not used by the system yet, it is recommended to keep all
    # the relevant configuration files in this directory.
    regression_training_configs: str

    # Classification Training Data Configs
    # Even though this path is not used by the system yet, it is recommended to keep all
    # the relevant configuration files in this directory.
    classification_training_data_configs: str

    # Classification Training Configs
    # Even though this path is not used by the system yet, it is recommended to keep all
    # the relevant configuration files in this directory.
    classification_training_configs: str








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

    # Filters
    filters: Union[List[int], None]

    # Kernel Sizes
    kernel_sizes: Union[List[int], None]

    # Pool Sizes
    pool_sizes: Union[List[int], None]

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







## Training History ##



# Training History
# The dictionary built once the training is completed. The properties adapt accordingly 
# based on the loss and metric functions used.
class IKerasModelTrainingHistory(TypedDict):
    # Training and validation loss
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
    rho: Union[str, None]
    momentum: Union[str, None]
    centered: Union[str, None]





# Model Loss Config
# The loss configuration used when the model was compiled. Keep in mind that
# all these values are stringified to ensure compatibility with the JSON file format.
class IKerasModelLossConfig(TypedDict):
    name: str
    reduction: Union[str, None]
    from_logits: Union[str, None]
    label_smoothing: Union[str, None]
    axis: Union[str, None]




# Model Layer
# A layer stacked with other layers within the model.
class IKerasModelLayer(TypedDict):
    name: str
    params: int
    input_shape: Tuple[Union[int, None]]
    output_shape: Tuple[Union[int, None]]
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
    input_shape: Tuple[Union[int, None]]
    output_shape: Tuple[Union[int, None]]

    # Layers
    layers: List[IKerasModelLayer]

    # Params
    total_params: int
    trainable_params: int
    non_trainable_params: int










