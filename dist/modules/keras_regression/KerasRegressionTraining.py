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
    IKerasRegressionTrainingConfig, IKerasRegressionTrainingCertificate, IModelEvaluation, IKerasOptimizer,\
        IKerasRegressionLoss, IKerasRegressionMetric, ITrainableModelType, IDiscovery, IDiscoveryPayload
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.regression_training_data.RegressionTrainingData import make_datasets
from modules.keras_models.KerasModel import KerasModel
from modules.keras_models.LearningRateSchedule import LearningRateSchedule
from modules.keras_models.KerasTrainingProgressBar import KerasTrainingProgressBar
from modules.keras_models.KerasModelSummary import get_summary
from modules.model.ModelType import validate_id
from modules.keras_regression.KerasRegression import KerasRegression
from modules.discovery.RegressionDiscovery import discover
from modules.model.KerasRegressionModel import KerasRegressionModel
from modules.model_evaluation.ModelEvaluation import evaluate



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
        "decay_rate": 0.35,
        "epochs": 100,
        "patience": 10,
        "batch_size": 32
    }






    ## Initialization ##




    def __init__(self, config: IKerasRegressionTrainingConfig, test_mode: bool=False):
        """Initializes the RegressionTraining Instance.

        Args:
            config: IKerasRegressionTrainingConfig
                The configuration that will be used to train the model.
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
        
        # Initialize the Keras Model's Configuration and populate the lookback
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["autoregressive"] = self.autoregressive
        self.keras_model["lookback"] = self.lookback
        self.keras_model["predictions"] = self.predictions

        # Make the datasets
        self.train_x, self.train_y, self.test_x, self.test_y = make_datasets(
            lookback=self.lookback,
            autoregressive=self.autoregressive,
            predictions=self.predictions,
            train_split=Epoch.TRAIN_SPLIT
        )

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
        print("    1/8) Initializing Model...")
        model: Sequential = KerasModel(config=self.keras_model)

        # Compile the model
        print("    2/8) Compiling Model...")
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[ self.metric ])
  
        # Train the model
        print("    3/8) Training Model")
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
            batch_size=KerasRegressionTraining.TRAINING_CONFIG["batch_size"],
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        print("    4/8) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.test_x, self.test_y, verbose=0) # [loss, metric]

        # Perform the regression discovery
        print("    5/8) Discovering KerasRegression")
        discovery, discovery_payload = discover(
            regression=KerasRegression(self.id, {
                "model": model,
                "autoregressive": self.autoregressive,
                "lookback": self.lookback,
                "predictions": self.predictions
            }),
            progress_bar_description="       ",
        )

        # Save the model
        print("    6/8) Saving Model...")
        self._save_model(model, discovery)

        # Perform the regression evaluation
        print("    7/8) Evaluating KerasRegressionModel")
        regression_model: KerasRegressionModel = KerasRegressionModel(
            config={
                "id": self.id,
                "keras_regressions": [ { "regression_id": self.id } ]
            },
            enable_cache=False
        )
        regression_evaluation: IModelEvaluation = evaluate(
            model=regression_model,
            price_change_requirement=discovery["successful_mean"],
            progress_bar_description="       ",
            discovery_completed=discovery_payload["early_stopping"] == None
        )

        # Save the certificate
        print("    8/8) Saving Certificate...")
        certificate: IKerasRegressionTrainingCertificate = self._save_certificate(
            start_time, 
            model, 
            history, 
            test_evaluation, 
            discovery,
            discovery_payload,
            regression_evaluation
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
            f.attrs["autoregressive"] = self.autoregressive
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
        discovery_payload: IDiscoveryPayload,
        regression_evaluation: IModelEvaluation
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
            regression_evaluation: IModelEvaluation
                The results of the regression post-training evaluation.

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
            discovery_payload,
            regression_evaluation
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
        discovery_payload: IDiscoveryPayload,
        regression_evaluation: IModelEvaluation
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
            regression_evaluation: IModelEvaluation
                The results of the regression post-training evaluation.

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
            "autoregressive": self.autoregressive,
            "lookback": self.lookback,
            "predictions": self.predictions,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_evaluation": test_evaluation,

            # Regression Discovery
            "discovery": discovery_payload,

            # Post Training Evaluation
            "regression_evaluation": regression_evaluation,

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