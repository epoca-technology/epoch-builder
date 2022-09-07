from typing import Union, List
from json import dumps
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import MeanSquaredError as MeanSquaredErrorMetric, MeanAbsoluteError as MeanAbsoluteErrorMetric
from keras.callbacks import EarlyStopping, History
from modules._types import IKerasTrainingTypeConfig, IKerasModelConfig, IKerasModelTrainingHistory,\
    IKerasRegressionTrainingConfig, IKerasRegressionTrainingCertificate, IKerasOptimizer, IKerasRegressionLoss, \
        IKerasRegressionMetric, ITrainableModelType, IDiscovery, IDiscoveryPayload
from modules._types.regression_training_data_types import IRegressionDatasets
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.keras_models.KerasModel import KerasModel
from modules.keras_models.LearningRateSchedule import LearningRateSchedule
from modules.keras_models.KerasTrainingProgress import KerasTrainingProgressBar, training_passed
from modules.keras_models.KerasModelSummary import get_summary
from modules.model.ModelType import validate_id
from modules.keras_regression.KerasRegression import KerasRegression
from modules.discovery.RegressionDiscovery import discover



class KerasRegressionTraining:
    """KerasRegressionTraining Class

    This class handles the training of KerasRegressions.

    Class Properties:
        TRAINING_CONFIG: IKerasTrainingTypeConfig
            The configuration to be used in order to train the models.

    Instance Properties:
        test_mode: bool
             If running from unit tests, it won't check the model's directory.
        model_type: ITrainableModelType
            The type of model that will be trained.
        id: str
            A descriptive identifier compatible with filesystems
        description: str
            Important information regarding the model that will be trained.
        autoregressive: bool
            The type of regression that will be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        learning_rate: float
            The learning rate that will be used to train the model.
        optimizer: Union[adam.Adam, rmsprop.RMSProp]                    "adam"|"rmsprop"
            The optimizer that will be used to train the model.
        loss: Union[MeanSquaredError, MeanAbsoluteError]                "mean_squared_error"|"mean_absolute_error"
            The loss function that will be used for training.
        metric: Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric]
            The metric function that will be used to evaluate the training.
        batch_size: int
            Number of samples per gradient update. Can be adjusted based on the network that will be trained.
        keras_model: IKerasModelConfig
            The configuration that will be used to build the Keras Model.
        train_x: ndarray
        train_y: ndarray
        test_x: ndarray
        test_y: ndarray
            Features and labels. The training data.
        train_size: int
            The number of rows included in the train dataset.
        test_size: int
            The number of rows included in the test dataset.
    """
    # Training Configuration
    TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "initial_lr": 0.01,
        "decay_steps": 1,
        "decay_rate": 0.1,
        "epochs": 200,
        "patience": 10,
        "batch_size": 32
    }






    ## Initialization ##




    def __init__(self, config: IKerasRegressionTrainingConfig, datasets: IRegressionDatasets, test_mode: bool=False):
        """Initializes the RegressionTraining Instance.

        Args:
            config: IKerasRegressionTrainingConfig
                The configuration that will be used to train the model.
            datasets: IRegressionDatasets
                The packed datasets that will be used to train and evaluate the
                regression.
            test_mode: bool
                If running from unit tests, it won't check the model's directory.

        Raises:
            ValueError:
                If the model is not correctly preffixed.
                If the model's directory already exists.
        """
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Init the type of model
        self.model_type: ITrainableModelType = "keras_regression"

        # Initialize the id
        validate_id("KerasRegressionModel", config["id"])
        self.id: str = config["id"]

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the type of regression
        self.autoregressive: bool = config["autoregressive"]

        # Initialize the lookback
        self.lookback: int = config["lookback"]

        # Initialize the predictions output
        self.predictions: int = config["predictions"]

        # Initialize the learning rate
        self.learning_rate: float = config["learning_rate"]

        # Initialize the optimizer function
        self.optimizer: Union[adam.Adam, rmsprop.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[MeanSquaredError, MeanAbsoluteError] = self._get_loss(config["loss"])

        # Initialize the metric function
        self.metric: Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric] = self._get_metric(config["metric"])

        # Initialize the batch size
        self.batch_size: int = self._get_batch_size()
        
        # Initialize the Keras Model's Configuration and populate the lookback
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["autoregressive"] = self.autoregressive
        self.keras_model["lookback"] = self.lookback
        self.keras_model["predictions"] = self.predictions

        # Make the datasets
        self.train_x, self.train_y, self.test_x, self.test_y = datasets

        # Initialize the Dataset Sizes
        self.train_size: int = self.train_x.shape[0]
        self.test_size: int = self.test_x.shape[0]

        # Make sure the model does not exist if not unit testing
        if not self.test_mode and Epoch.FILE.model_exists(self.id, self.model_type):
            raise RuntimeError(f"Cannot train the model {self.id} because it already exists.")









    def _get_optimizer(self, func_name: IKerasOptimizer) -> Union[adam.Adam, rmsprop.RMSProp]:
        """Based on a optimizer function name, it will return the instance ready to be initialized.

        Args:
            func_name: IKerasOptimizer
                The name of the optimizer function to be used.

        Returns:
            Union[adam_v2.Adam, rmsprop_v2.RMSProp]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        # Initialize the Learning Rate Schedule
        lr_schedule: Union[InverseTimeDecay, float] = LearningRateSchedule(
            learning_rate=self.learning_rate,
            initial_learning_rate=KerasRegressionTraining.TRAINING_CONFIG["initial_lr"],
            decay_steps=KerasRegressionTraining.TRAINING_CONFIG["decay_steps"],
            decay_rate=KerasRegressionTraining.TRAINING_CONFIG["decay_rate"]
        )

        # Return the Optimizer Instance
        if func_name == "adam":
            return adam.Adam(lr_schedule)
        elif func_name == "rmsprop":
            return rmsprop.RMSProp(lr_schedule)
        else:
            raise ValueError(f"The optimizer function for {func_name} was not found.")








    def _get_loss(self, func_name: IKerasRegressionLoss) -> Union[MeanSquaredError, MeanAbsoluteError]:
        """Based on a loss function name, it will return the instance ready to be initialized.

        Args:
            func_name: IKerasRegressionLoss
                The name of the loss function to be used.

        Returns:
            Union[MeanSquaredError, MeanAbsoluteError]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == "mean_squared_error":
            return MeanSquaredError()
        elif func_name == "mean_absolute_error":
            return MeanAbsoluteError()
        else:
            raise ValueError(f"The loss function for {func_name} was not found.")







    def _get_metric(self, func_name: IKerasRegressionMetric) -> Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric]:
        """Based on a metric function name, it will return the instance ready to be initialized.

        Args:
            func_name: IKerasRegressionMetric
                The name of the metric function to be used.

        Returns:
            Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == "mean_squared_error":
            return MeanSquaredErrorMetric()
        elif func_name == "mean_absolute_error":
            return MeanAbsoluteErrorMetric()
        else:
            raise ValueError(f"The metric function for {func_name} was not found.")






    def _get_batch_size(self) -> int:
        """Retrieves the batch size that will be used to train the models based
        on the network.

        Returns:
            int
        """
        if "DNN" in self.id:
            return 2
        elif "CNN" in self.id:
            return 16
        elif "CLSTM" in self.id:
            return 256
        elif "LSTM" in self.id:
            return 128
        else:
            return KerasRegressionTraining.TRAINING_CONFIG["batch_size"]



    



    ## Training ##






    def train(self) -> IKerasRegressionTrainingCertificate:
        """Compiles, trains and saves the model as well as the training certificate.

        Returns:
            IKerasRegressionTrainingCertificate
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            mode="min", 
            patience=KerasRegressionTraining.TRAINING_CONFIG["patience"],
            restore_best_weights=True
        )

        # Retrieve the Keras Model
        print("    1/7) Initializing Model...")
        model: Sequential = KerasModel(config=self.keras_model)

        # Compile the model
        print("    2/7) Compiling Model...")
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[ self.metric ])
  
        # Train the model
        print("    3/7) Training Model")
        history_object: History = model.fit(
            self.train_x, 
            self.train_y, 
            validation_split=0.2, 
            epochs=KerasRegressionTraining.TRAINING_CONFIG["epochs"],
            callbacks=[ 
                early_stopping, 
                KerasTrainingProgressBar(KerasRegressionTraining.TRAINING_CONFIG["epochs"], "       ") 
            ],
            shuffle=True,
            batch_size=self.batch_size,
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        print("    4/7) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.test_x, self.test_y, verbose=0) # [loss, metric]

        # Perform the regression discovery
        print("    5/7) Discovering KerasRegression")
        discovery, discovery_payload = discover(
            regression=KerasRegression(self.id, {
                "model": model,
                "autoregressive": self.autoregressive,
                "lookback": self.lookback,
                "predictions": self.predictions
            }),
            progress_bar_description="       ",
            training_passed=training_passed(history, KerasRegressionTraining.TRAINING_CONFIG["epochs"])
        )

        # Save the model
        print("    6/7) Saving Model...")
        self._save_model(model, discovery)

        # Save the certificate
        print("    7/7) Saving Certificate...")
        certificate: IKerasRegressionTrainingCertificate = self._save_certificate(
            start_time, 
            model, 
            history, 
            test_evaluation, 
            discovery,
            discovery_payload
        )

        # Return it so it can be added to the batch
        return certificate













    ## Trained Model Saving ##






    def _save_model(self, model: Sequential, discovery: IDiscovery) -> None:
        """Saves a trained model in the output directory as well as the training certificate.

        Args:
            model: Sequential
                The instance of the trained model.
            discovery: IDiscovery
                The result of the regression discovery.
        """
        # Create the model's directory
        Epoch.FILE.make_active_model_dir(self.id)
        
        # Save the model with the required metadata
        with h5pyFile(Epoch.FILE.get_active_model_path(self.id, self.model_type), mode="w") as f:
            save_model_to_hdf5(model, f)
            f.attrs["id"] = self.id
            f.attrs["description"] = self.description
            f.attrs["lookback"] = self.lookback
            f.attrs["predictions"] = self.predictions
            f.attrs["discovery"] = dumps(discovery)







    def _save_certificate(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        discovery: IDiscovery,
        discovery_payload: IDiscoveryPayload
    ) -> IKerasRegressionTrainingCertificate:
        """Saves a trained model in the output directory as well as the training certificate.

        Args:
            start_time: int
                The time in which the training started.
            model: Sequential
                The instance of the trained model.
            training_history: IKerasModelTrainingHistory
                The dictionary containing the training history.
            test_evaluation: List[float]
                The results when evaluating the test dataset.
            discovery: IDiscovery
            discovery_payload: IDiscoveryPayload
                The discovery and the payload of the regression.

        Returns:
            IKerasRegressionTrainingCertificate
        """
        # Build the certificate
        certificate: IKerasRegressionTrainingCertificate = self._get_certificate(
            model, 
            start_time, 
            training_history, 
            test_evaluation, 
            discovery,
            discovery_payload
        )

        # Save the file
        Epoch.FILE.save_training_certificate(certificate)

        # Finally, return it so it can be added to the batch
        return certificate







    def _get_certificate(
        self,
        model: Sequential,
        start_time: int, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        discovery: IDiscovery,
        discovery_payload: IDiscoveryPayload
    ) -> IKerasRegressionTrainingCertificate:
        """Builds the certificate that contains all the data regarding the training process
        that will be saved alongside the model.

        Args:
            model: Sequential
                The trained model which will provide the summary.
            start_time: int
                The time in which the training started.
            training_history: IKerasModelTrainingHistory
                The model's performance history during training.
            test_evaluation: List[float]
                The evaluation performed on the test dataset.
            discovery: IDiscovery
            discovery_payload: IDiscoveryPayload
                The discovery and the payload of the regression.

        Returns:
            IKerasRegressionTrainingCertificate
        """
        return {
            # Identification
            "id": self.id,
            "description": self.description,

            # Training Data
            "training_data_start": int(Candlestick.NORMALIZED_PREDICTION_DF.iloc[0]["ot"]),
            "training_data_end": int(Candlestick.NORMALIZED_PREDICTION_DF.iloc[-1]["ct"]),
            "train_size": self.train_size,
            "test_size": self.test_size,
            "training_data_summary": Candlestick.NORMALIZED_PREDICTION_DF["c"].describe().to_dict(),

            # Training Configuration
            "lookback": self.lookback,
            "predictions": self.predictions,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "metric": self.metric.name,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_evaluation": test_evaluation,

            # Regression Discovery
            "discovery": discovery_payload,

            # The configuration of the Regression
            "regression_config": {
                "id": self.id,
                "description": self.description,
                "autoregressive": self.autoregressive,
                "lookback": self.lookback,
                "predictions": self.predictions,
                "discovery": discovery,
                "summary": get_summary(model)
            }
        }