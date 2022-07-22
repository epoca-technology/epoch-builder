from typing import Literal, TypedDict, Union, List, Tuple







## Configuration ##


# Optimizer Functions
IKerasOptimizer = Literal["adam", "rmsprop"]


# Loss Functions
IKerasLoss = Literal["mean_absolute_error", "mean_squared_error", "categorical_crossentropy", "binary_crossentropy"]
IKerasRegressionLoss = Literal["mean_absolute_error", "mean_squared_error"]
IKerasClassificationLoss = Literal["categorical_crossentropy", "binary_crossentropy"]


# Metric Functions
IKerasMetric = Literal["mean_absolute_error", "mean_squared_error", "categorical_accuracy", "binary_accuracy"]
IKerasRegressionMetric = Literal["mean_absolute_error", "mean_squared_error"]
IKerasClassificationMetric = Literal["categorical_accuracy", "binary_accuracy"]


# Activation Functions
IKerasActivation = Literal["relu", "tanh"]



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
    activations: Union[List[str], List[IKerasActivation], None]

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







## Training Configuration ##



# Keras Model Training Type Configuration
# Based on the type of training (hyperparams|shortlist), different training settings will be used.
# For more information regarding these args, view the KerasTraining.ipynb notebook.
class IKerasTrainingTypeConfig(TypedDict):
    # The split that will be applied to the dataset that will generate the train and test datasets
    train_split: float

    # A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
    initial_lr: float

    # How often to apply decay.
    decay_steps: float

    # A Python number. The decay rate for the learning rate per step.
    decay_rate: float

    # The maximum number of epochs the training process will go through
    epochs: int

    # Number of epochs with no improvement after which training will be stopped.
    patience: int

    # Number of samples per gradient update. If unspecified, batch_size will default to 32. 
    # Do not specify the batch_size if your data is in the form of datasets
    batch_size: int










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



# Optimizer Name
IKerasOptimizerName = Literal["Adam", "RMSprop"]



# Model Optimizer Config
# The optimizer configuration used when the model was compiled. Keep in mind that
# all these values are stringified to ensure compatibility with the JSON file format.
class IKerasModelOptimizerConfig(TypedDict):
    name: IKerasOptimizerName
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
    name: IKerasLoss
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




# Model Class Name
IKerasModelClassName = Literal["Sequential"]


# Metric Names
IKerasMetricName = Literal["loss", "categorical_accuracy", "binary_accuracy"]


# Model Summary
# Extracts all the relevant information from a trained model.
class IKerasModelSummary(TypedDict):
    # The name of the class used to create the model.
    model_class: IKerasModelClassName

    # Optimizer Config
    optimizer_config: IKerasModelOptimizerConfig

    # Loss Config
    loss_config: IKerasModelLossConfig

    # Metrics
    metrics: List[IKerasMetricName]

    # Input and output shapes
    input_shape: Tuple[Union[int, None]]
    output_shape: Tuple[Union[int, None]]

    # Layers
    layers: List[IKerasModelLayer]

    # Params
    total_params: int
    trainable_params: int
    non_trainable_params: int










