from typing import Union, Tuple, List, Dict, Any
from os import makedirs
from os.path import exists
from pandas import DataFrame
from json import dumps
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import MeanSquaredError as MeanSquaredErrorMetric, MeanAbsoluteError as MeanAbsoluteErrorMetric
from keras.callbacks import EarlyStopping, History
from keras.optimizers import adam_v2, rmsprop_v2
from h5py import File as h5pyFile
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.keras_models import KerasModel, IKerasModelConfig, IKerasModelTrainingHistory, get_summary
from modules.regression import IRegressionTrainingConfig, TrainingWindowGenerator, IRegressionTrainingCertificate




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
            A descriptive identifier compatible with filesystems
        description: str
            Important information regarding the model that will be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        learning_rate: float
            The learning rate to be used by the optimizer. If None is provided it uses the default
            one.   
        optimizer: Union[adam_v2.Adam, rmsprop_v2.RMSProp]
            The optimizer that will be used to train the model.
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
        train_size: int
            The number of rows included in the train dataset.
        val_size: int
            The number of rows included in the val dataset.
        test_size: int
            The number of rows included in the test dataset.
    """

    # Directory where the model and the training certificate will be stored
    OUTPUT_PATH: str = './saved_keras_models'

    # The maximum number of EPOCHs a model can go through during training
    MAX_EPOCHS: int = 200
    #MAX_EPOCHS: int = 1




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
                If the model is not correctly preffixed.
                If the model's directory already exists.
        """
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the id
        self.id: str = config['id']
        if self.id[0:2] != 'R_':
            raise ValueError("The ID of the Regression Model must be preffixed with R_")

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

        # Initialize the Keras Model's Configuration and populate the lookback and the # of predictions
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["lookback"] = self.lookback
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

        # Initialize the Dataset Sizes
        self.train_size: int = train_df.shape[0]
        self.val_size: int = val_df.shape[0]
        self.test_size: int = test_df.shape[0]

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






    def run(self) -> None:
        """Compiles, trains and saves the model as well as the training certificate.
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        # Retrieve the Keras Model
        model: Sequential = KerasModel(model_type='regression', config=self.keras_model)

        # Compile the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
  
        # Train the model
        print(f"Training {self.id}...\n")
        history_object: History = model.fit(
            self.window.train, 
            epochs=RegressionTraining.MAX_EPOCHS,
            validation_data=self.window.val,
            callbacks=[ early_stopping ]
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        print(f"\nEvaluating test data...")
        test_evaluation: List[float] = model.evaluate(self.window.test) # [loss, metric]

        # Finally, save the model
        self._save_model(start_time, model, history, test_evaluation)












    ## Trained Model Handling ##


    def _save_model(
        self, 
        start_time: int, 
        model: Union[Sequential, Any], 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float]
    ) -> None:
        """Saves a trained model in the output directory as well as the training certificate.

        Args:
            start_time: int
                The time in which the training started.
            model: ...
                The instance of the trained model.
            training_history: IKerasModelTrainingHistory
                The dictionary containing the training history.
            test_evaluation: List[float]
                The results when evaluating the test dataset.
        """
        # Create the model's directory
        makedirs(f"{RegressionTraining.OUTPUT_PATH}/{self.id}")
        
        # Save the model with the required metadata
        with h5pyFile(f"{RegressionTraining.OUTPUT_PATH}/{self.id}/model.h5", mode='w') as f:
            save_model_to_hdf5(model, f)
            f.attrs['id'] = self.id
            f.attrs['lookback'] = self.lookback
            f.attrs['predictions'] = self.predictions

        # Save the certificate file
        with open(f"{RegressionTraining.OUTPUT_PATH}/{self.id}/certificate.json", "w") as outfile:
            outfile.write(dumps(self._get_certificate(model, start_time, training_history, test_evaluation)))






    def _get_certificate(
        self,
        model: Union[Sequential, Any],
        start_time: int, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float]
    ) -> IRegressionTrainingCertificate:
        """Builds the certificate that contains all the data regarding the training process
        that will be saved alongside the model.

        Args:
            model: ...
                The trained model which will provide the summary.
            start_time: int
                The time in which the training started.
            training_history: IKerasModelTrainingHistory
                The model's performance history during training.
            test_evaluation: List[float]
                The evaluation performed on the test dataset.

        Returns:
            IRegressionTrainingCertificate
        """
        return {
            # Identification
            "id": self.id,
            "description": self.description,

            # Training Data
            "training_data_start": int(Candlestick.PREDICTION_DF.iloc[0]['ot']),
            "training_data_end": int(Candlestick.PREDICTION_DF.iloc[-1]['ct']),
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "training_data_summary": Candlestick.NORMALIZED_PREDICTION_DF.describe().to_dict(),

            # Training Configuration
            "lookback": self.lookback,
            "predictions": self.predictions,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "metric": self.metric.name,
            "batch_size": self.batch_size,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_evaluation": test_evaluation,
            "keras_model_summary": get_summary(model, include_weights=True),
        }









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
        if exists(f"{RegressionTraining.OUTPUT_PATH}/{self.id}"):
            raise ValueError(f"The model {self.id} already exists.")