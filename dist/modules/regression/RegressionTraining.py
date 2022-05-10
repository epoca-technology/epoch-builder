from os import makedirs
from os.path import exists
from typing import Union, Tuple, List, Dict
from pandas import DataFrame
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import MeanSquaredError as MeanSquaredErrorMetric, MeanAbsoluteError as MeanAbsoluteErrorMetric
from keras.callbacks import EarlyStopping, History
from keras.optimizers import adam_v2, rmsprop_v2
from h5py import File as h5pyFile
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.keras_models import KerasRegressionModel, IKerasModelConfig
from modules.regression import IRegressionTrainingConfig, TrainingWindowGenerator, IRegressionTrainingHistory, \
    IRegressionTrainingCertificate




class RegressionTraining:
    """RegressionTraining Class

    This class handles the training of a Forecasting Model.

    Class Properties:
        OUTPUT_PATH: str
            The directory in which the models will be stored.
        MAX_EPOCHS: int
            The maximum amount of epochs the training process can go through.
        DEFAULT_LR: float
            The default learning rate to be used by the optimizer if none is provided.

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks.
        id: str
            A uuid that is generated once the instance is initialized.
        description: str
            Important information regarding the model that will be trained.
        name: str
            The name of the model about to be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        optimizer: Union[adam_v2.Adam, rmsprop_v2.RMSProp]
            The optimizer that will be used to train the model.
        learning_rate: float
            The learning rate to be used by the optimizer. If None is provided it uses the default
            one.   
        loss: Union[MeanSquaredError, MeanAbsoluteError]
            The loss function that will be used for training.
        metric: Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric]
            The metric function that will be used for training.
        batch_size: int
            The size of the training dataset batches.
        keras_model: IKerasModelConfig
            The configuration that will be used to build the Keras Model.
        window: ForecastingTrainingWindowGenerator
            The instance of the Window Generator
    """

    # Directory where the model and the training certificate will be stored
    OUTPUT_PATH: str = './regression_models'

    # The maximum number of EPOCHs a model can go through during training
    MAX_EPOCHS: int = 200




    ## Initialization ##




    def __init__(self, config: IRegressionTrainingConfig, test_mode: bool = False):
        """Initializes the ForecastingTraining Instance.

        Args:
            config: IRegressionTrainingConfig
                The configuration that will be used to train the model.
            test_mode: bool
                Indicates if the execution is running from unit tests.

        Raises:
            ValueError:
                If the model's directory already exists.
        """
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the Identifier
        self.id: str = Utils.generate_uuid4()

        # Initialize the name
        self.name: str = config['name']

        # Initialize the description
        self.description: str = config['description']

        # Initialize the lookback
        self.lookback: int = config['lookback']

        # Initialize the predictions output
        self.predictions: int = config['predictions']

        # Initialize the Learning Rate
        self.learning_rate: float = config['learning_rate']

        # Initialize the optimizer function
        self.optimizer: Union[adam_v2.Adam, rmsprop_v2.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[MeanSquaredError, MeanAbsoluteError] = self._get_loss(config['loss'])

        # Initialize the metric function
        self.metric: Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric] = self._get_metric(config['metric'])

        # Initialize the Batch Size
        self.batch_size: int = config["batch_size"]

        # Initialize the Keras Model's Configuration and populate the # of predictions
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["predictions"] = self.predictions

        # Initialize the candlesticks if not unit testing
        if not self.test_mode:
            Candlestick.init(self.lookback, normalized_df=True)

        # Split the candlesticks into train, val and test
        train_df, val_df, test_df = self._get_data()

        # Initialize the Window Instance
        self.window: TrainingWindowGenerator = TrainingWindowGenerator({
            "input_width": self.lookback,
            "label_width": self.predictions,
            "shift": self.predictions,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "label_columns": ["c"],
            "batch_size": self.batch_size
        })

        # Initialize the model's directory
        self._init_model_dir()





    

    def _get_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the normalized prediction candlesticks into train, val and test dataframes.

        Returns:
            Tuple[DataFrame, DataFrame, DataFrame] 
            (train_df, val_df, test_df)
        """
        # Initialize the total rows
        rows: int = Candlestick.NORMALIZED_PREDICTION_DF.shape[0]

        # Split the DataFrames
        train_df: DataFrame = Candlestick.NORMALIZED_PREDICTION_DF[0:int(rows * 0.7)]
        val_df: DataFrame = Candlestick.NORMALIZED_PREDICTION_DF[int(rows * 0.7):int(rows * 0.9)]
        test_df: DataFrame = Candlestick.NORMALIZED_PREDICTION_DF[int(rows * 0.9):]

        # Return the packed DataFrames
        return train_df, val_df, test_df






    def _get_optimizer(self, func_name: str) -> Union[adam_v2.Adam, rmsprop_v2.RMSProp]:
        """Based on a optimizer function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the optimizer function to be used.

        Returns:
            Union[adam_v2.Adam, rmsprop_v2.RMSProp]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == 'adam':
            return adam_v2.Adam(learning_rate=self.learning_rate)
        elif func_name == 'rmsprop':
            return rmsprop_v2.RMSProp(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"The optimizer function for {func_name} was not found.")






    def _get_loss(self, func_name: str) -> Union[MeanSquaredError, MeanAbsoluteError]:
        """Based on a loss function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the loss function to be used.

        Returns:
            Union[MeanSquaredError, MeanAbsoluteError]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == 'mse':
            return MeanSquaredError()
        elif func_name == 'mae':
            return MeanAbsoluteError()
        else:
            raise ValueError(f"The loss function for {func_name} was not found.")







    def _get_metric(self, func_name: str) -> Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric]:
        """Based on a metric function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the loss function to be used.

        Returns:
            Union[MeanSquaredErrorMetric, MeanSquaredErrorMetric]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == 'mse':
            return MeanSquaredErrorMetric()
        elif func_name == 'mae':
            return MeanAbsoluteErrorMetric()
        else:
            raise ValueError(f"The metric function for {func_name} was not found.")






    



    ## Training ##






    def train(self) -> None:
        """Compiles, trains and saves the model as well as the training certificate.
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        # Retrieve the Keras Model
        model: Sequential = KerasRegressionModel(self.keras_model)

        # Compile the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
  
        # Train the model
        print(f"\nTraining Model")
        print(f"ID: {self.id}")
        print(f"Name: {self.name}\n")
        history_object: History = model.fit(
            self.window.train, 
            epochs=RegressionTraining.MAX_EPOCHS,
            validation_data=self.window.val,
            callbacks=[ early_stopping ]
        )

        # Initialize the Training History
        history: IRegressionTrainingHistory = history_object.history

        # Evaluate the test dataset
        print(f"\nEvaluating test data...")
        test_evaluation: List[float, float] = model.evaluate(self.window.test) # [loss, metric]

        print("\nHistory: ", history)
        print("\nTest Data Evaluation: ", test_evaluation)

        # Save the model with the required metadata
        with h5pyFile(f"{RegressionTraining.OUTPUT_PATH}/{self.name}/model.h5", mode='w') as f:
            save_model_to_hdf5(model, f)
            f.attrs['id'] = self.id
            f.attrs['lookback'] = self.lookback
            f.attrs['predictions'] = self.predictions

        # Save the Model and the Certificate
        # @TODO











    ## Trained Model Handling ##


    def _save_model(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IRegressionTrainingHistory, 
        test_evaluation: List[float, float]
    ) -> None:
        """Saves a trained model in the output directory as well as the training certificate.

        Args:
            start_time: int
                The time in which the training started.
            model: Sequential
                The instance of the trained model.
            training_history: IRegressionTrainingHistory
                The dictionary containing the training history.
            test_evaluation: List[float, float]
                The results when evaluating the test dataset.
        """
        pass







    def _get_certificate(
        self,
        start_time: int, 
        training_history: IRegressionTrainingHistory, 
        test_evaluation: List[float, float]
    ) -> IRegressionTrainingCertificate:
        """Builds the certificate that contains all the data regarding the training process
        that will be saved alongside the model.

        Args:
            start_time: int
                The time in which the training started.
            
        """
        # Initialize the end time
        end_time: int = Utils.get_time()

        # Initialize the Train Data Summary
        train_data_summary: Dict[str, Dict[str, float]] = Candlestick.NORMALIZED_PREDICTION_DF.describe().to_dict()










    def _init_model_dir(self) -> None:
        """Creates the required directories in which the model and the certificate
        will be stored.

        Raises:
            ValueError:
                If the model's directory already exists.
        """
        # If the output directory doesn't exist, create it
        if not exists(RegressionTraining.OUTPUT_PATH):
            makedirs(RegressionTraining.OUTPUT_PATH)

        # Make sure the model's directory does not exist
        if exists(f"{RegressionTraining.OUTPUT_PATH}/{self.name}"):
            raise ValueError(f"The model directory {self.name} already exists.")
        else:
            makedirs(f"{RegressionTraining.OUTPUT_PATH}/{self.name}")