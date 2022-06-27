from typing import Union, Tuple, List
from os import makedirs
from os.path import exists
from numpy import ndarray, array
from json import dumps
from h5py import File as h5pyFile
from tensorflow import random as tf_random
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping, History
from modules.types import IKerasTrainingTypeConfig, IKerasModelConfig, IKerasModelTrainingHistory,\
    IRegressionTrainingConfig, IRegressionTrainingCertificate, IModelEvaluation
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.keras_models.KerasPath import KERAS_PATH
from modules.keras_models.KerasModel import KerasModel
from modules.keras_models.LearningRateSchedule import LearningRateSchedule
from modules.keras_models.KerasModelSummary import get_summary
from modules.model_evaluation.ModelEvaluation import evaluate




class RegressionTraining:
    """RegressionTraining Class

    This class handles the training of a RegressionModel.

    Class Properties:
        HYPERPARAMS_TRAINING_CONFIG: IKerasTrainingTypeConfig
        SHORTLISTED_TRAINING_CONFIG: IKerasTrainingTypeConfig
            The configurations to be used based on the type of training.

    Instance Properties:
        test_mode: bool
             If running from unit tests, it won't check the model's directory.
        hyperparams_mode: bool
            If enabled, it means that the purpose of the training is to identify the best hyperparams
            and therefore, a large amount of models will be trained.
        training_config: IRegressionTrainingTypeConfig
            The config of the type of training that will be performed (Hyperparams|Shortlisted).
        id: str
            A descriptive identifier compatible with filesystems
        model_path: str
            The directory in which the model will be stored.
        description: str
            Important information regarding the model that will be trained.
        autoregressive: bool
            The type of regression that will be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        optimizer: Union[adam.Adam, rmsprop.RMSProp]                    "adam"|"rmsprop"
            The optimizer that will be used to train the model.
        loss: Union[MeanSquaredError, MeanAbsoluteError]                "mean_squared_error"|"mean_absolute_error"
            The loss function that will be used for training.
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
    # Hyperparams Training Configuration
    HYPERPARAMS_TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "initial_lr": 0.01,
        "decay_steps": 1.5,
        "decay_rate": 0.55,
        "epochs": 50,
        "patience": 10
    }

    # Shortlisted Training Configuration
    SHORTLISTED_TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "initial_lr": 0.01,
        "decay_steps": 2,
        "decay_rate": 0.065,
        "epochs": 500,
        "patience": 50
    }






    ## Initialization ##




    def __init__(
        self, 
        config: IRegressionTrainingConfig, 
        hyperparams_mode: bool=False,
        datasets: Union[Tuple[ndarray, ndarray, ndarray, ndarray], None]=None,
        test_mode: bool=False
    ):
        """Initializes the RegressionTraining Instance.

        Args:
            config: IRegressionTrainingConfig
                The configuration that will be used to train the model.
            hyperparams_mode: bool
                If enabled, there will be no verbosity during training and eval.
            test_mode: bool
                If running from unit tests, it won't check the model's directory.

        Raises:
            ValueError:
                If the model is not correctly preffixed.
                If the model's directory already exists.
        """
        # Set the Global Random Seed to ensure training reproducibility
        tf_random.set_seed(60184)

        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the mode
        self.hyperparams_mode: bool = hyperparams_mode

        # Set the type of training that will be performed
        self.training_config: IKerasTrainingTypeConfig = \
            RegressionTraining.HYPERPARAMS_TRAINING_CONFIG if self.hyperparams_mode \
                else RegressionTraining.SHORTLISTED_TRAINING_CONFIG

        # Initialize the id
        self.id: str = config['id']
        if self.id[0:2] != 'R_':
            raise ValueError("The ID of the Regression Model must be preffixed with R_")

        # Initialize the Model's path
        self.model_path: str = f"{KERAS_PATH['models']}/{self.id}"

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the type of regression
        self.autoregressive: bool = config["autoregressive"]

        # Initialize the lookback
        self.lookback: int = config["lookback"]

        # Initialize the predictions output
        self.predictions: int = config["predictions"]

        # Initialize the optimizer function
        self.optimizer: Union[adam.Adam, rmsprop.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[MeanSquaredError, MeanAbsoluteError] = self._get_loss(config['loss'])

        # Initialize the Keras Model's Configuration and populate the lookback
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["autoregressive"] = self.autoregressive
        self.keras_model["lookback"] = self.lookback
        self.keras_model["predictions"] = self.predictions

        # Make the datasets if they weren't provided
        if datasets is None:
            self.train_x, self.train_y, self.test_x, self.test_y = RegressionTraining.make_datasets(
                lookback=self.lookback,
                autoregressive=self.autoregressive,
                predictions=self.predictions
            )

        # Otherwise, unpack the provided datasets
        else:
            self.train_x, self.train_y, self.test_x, self.test_y = datasets

        # Initialize the Dataset Sizes
        self.train_size: int = self.train_x.shape[0]
        self.test_size: int = self.test_x.shape[0]

        # Initialize the model's directory if not unit testing
        if not self.test_mode:
            self._init_model_dir()







    def _get_optimizer(self, func_name: str) -> Union[adam.Adam, rmsprop.RMSProp]:
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
        # Initialize the Learning Rate Schedule
        lr_schedule: InverseTimeDecay = LearningRateSchedule(
            initial_learning_rate=self.training_config["initial_lr"],
            decay_steps=self.training_config["decay_steps"],
            decay_rate=self.training_config["decay_rate"]
        )

        # Return the Optimizer Instance
        if func_name == 'adam':
            return adam.Adam(lr_schedule)
        elif func_name == 'rmsprop':
            return rmsprop.RMSProp(lr_schedule)
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
        if func_name == 'mean_squared_error':
            return MeanSquaredError()
        elif func_name == 'mean_absolute_error':
            return MeanAbsoluteError()
        else:
            raise ValueError(f"The loss function for {func_name} was not found.")








    
    @staticmethod
    def make_datasets(lookback: int, autoregressive: bool, predictions: int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Builds a tuple containing the features and labels for the train and test datasets based
        on the kind of regression. 

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]
            (train_x, train_y, test_x, test_y)
        """
        # Init the number of rows and the split that will be applied
        rows: int = Candlestick.NORMALIZED_PREDICTION_DF.shape[0]
        split: int = int(rows * 0.8)

        # Init raw features and labels
        features_raw: Union[List[List[float]], ndarray] = []
        labels_raw: Union[List[List[float]], ndarray] = []

        # Iterate over the normalized ds and build the features & labels
        for i in range(lookback, rows):
            # If it is an autoregression, add only 1 price as the label
            if autoregressive:
                features_raw.append(Candlestick.NORMALIZED_PREDICTION_DF.iloc[i-lookback:i, 0])
                labels_raw.append(Candlestick.NORMALIZED_PREDICTION_DF.iloc[i, 0])

            # If it is not an autoregression, add the labels based on the number of predictions
            elif not autoregressive and i < (rows-predictions):
                features_raw.append(Candlestick.NORMALIZED_PREDICTION_DF.iloc[i-lookback:i, 0])
                labels_raw.append(Candlestick.NORMALIZED_PREDICTION_DF.iloc[i:i+predictions, 0])

        # Convert the features and labels into np arrays
        features = array(features_raw)
        labels = array(labels_raw)

        # Finally, return the split datasets
        return features[:split], labels[:split], features[split:], labels[split:]








    



    ## Training ##






    def train(self) -> IRegressionTrainingCertificate:
        """Compiles, trains and saves the model as well as the training certificate.

        Returns:
            IRegressionTrainingCertificate
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=self.training_config["patience"],
            restore_best_weights=True
        )

        # Retrieve the Keras Model
        if not self.hyperparams_mode:
            print("    1/7) Initializing Model...")
        model: Sequential = KerasModel(config=self.keras_model)

        # Compile the model
        if not self.hyperparams_mode:
            print("    2/7) Compiling Model...")
        model.compile(optimizer=self.optimizer, loss=self.loss)
  
        # Train the model
        if not self.hyperparams_mode:
            print("    3/7) Training Model...")
        history_object: History = model.fit(
            self.train_x, 
            self.train_y, 
            validation_split=0.2, 
            epochs=self.training_config["epochs"],
            callbacks=[ early_stopping ],
            shuffle=True,
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        if not self.hyperparams_mode:
            print("    4/7) Evaluating Test Dataset...")
        test_evaluation: float = model.evaluate(self.test_x, self.test_y, verbose=0) # loss

        # Save the model
        if not self.hyperparams_mode:
            print("    5/7) Saving Model...")
        self._save_model(model)

        # Perform the regression evaluation
        regression_evaluation: IModelEvaluation = self._evaluate_regression()

        # Save the certificate
        if not self.hyperparams_mode:
            print("    7/7) Saving Certificate...")
        certificate: IRegressionTrainingCertificate = self._save_certificate(
            start_time, 
            model, 
            history, 
            test_evaluation, 
            regression_evaluation
        )

        # Return it so it can be added to the batch
        return certificate









    def _evaluate_regression(self) -> IModelEvaluation:
        """Loads the trained model that has just been saved and performs a runs
        an evaluation that is similar to the backtest. This eval will only run
        on the test dataset

        Returns:
            IModelEvaluation
        """
        # Price Change Requirement
        # This value is set based on the best combinations in the regression selection.
        # So far we know there are better chances of succeeding in the 2.5-3 range.
        price_change_requirement: float = 3

        # Init the number of rows and the split that will be applied
        rows: int = Candlestick.PREDICTION_DF.shape[0]
        split: int = int(rows * 0.8)

        # Initialize the first open time of the test dataset
        first_ot: int = Candlestick.PREDICTION_DF[split:split+1].iloc[0]["ot"]

        # Finally, evaluate the model
        return evaluate(
            model_config={
                "id": self.id,
                "regression_models": [ {"regression_id": self.id, "interpreter": { "long": 1, "short": 1 }} ]
            },
            start_timestamp=first_ot,
            price_change_requirement=price_change_requirement,
            hyperparams_mode=self.hyperparams_mode
        )
















    ## Trained Model Saving ##






    def _save_model(self, model: Sequential) -> None:
        """Saves a trained model in the output directory as well as the training certificate.

        Args:
            model: Sequential
                The instance of the trained model.
        """
        # Create the model's directory
        makedirs(self.model_path)
        
        # Save the model with the required metadata
        with h5pyFile(f"{self.model_path}/model.h5", mode='w') as f:
            save_model_to_hdf5(model, f)
            f.attrs['id'] = self.id
            f.attrs['description'] = self.description
            f.attrs['autoregressive'] = self.autoregressive
            f.attrs['lookback'] = self.lookback
            f.attrs['predictions'] = self.predictions







    def _save_certificate(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        regression_evaluation: IModelEvaluation
    ) -> IRegressionTrainingCertificate:
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
            regression_evaluation: IModelEvaluation
                The results of the regression post-training evaluation.

        Returns:
            IRegressionTrainingCertificate
        """
        # Build the certificate
        certificate: IRegressionTrainingCertificate = self._get_certificate(
            model, 
            start_time, 
            training_history, 
            test_evaluation, 
            regression_evaluation
        )

        # Save the file
        with open(f"{self.model_path}/certificate.json", "w") as outfile:
            outfile.write(dumps(certificate))

        # Finally, return it so it can be added to the batch
        return certificate







    def _get_certificate(
        self,
        model: Sequential,
        start_time: int, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        regression_evaluation: IModelEvaluation
    ) -> IRegressionTrainingCertificate:
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
            regression_evaluation: IModelEvaluation
                The results of the regression post-training evaluation.

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
            "test_size": self.test_size,
            "training_data_summary": Candlestick.NORMALIZED_PREDICTION_DF["c"].describe().to_dict(),

            # Training Configuration
            "autoregressive": self.autoregressive,
            "lookback": self.lookback,
            "predictions": self.predictions,
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_evaluation": test_evaluation,

            # Post Training Evaluation
            "regression_evaluation": regression_evaluation,

            # The configuration of the Regression
            "regression_config": {
                "id": self.id,
                "description": self.description,
                "autoregressive": self.autoregressive,
                "lookback": self.lookback,
                "predictions": self.predictions,
                "summary": get_summary(model)
            }
        }









    def _init_model_dir(self) -> None:
        """Creates the required directories in which the model and the certificate
        will be stored.

        Raises:
            ValueError:
                If the model's directory already exists.
        """
        # If the output directory doesn't exist, create it
        if not exists(KERAS_PATH["models"]):
            makedirs(KERAS_PATH["models"])

        # Make sure the model's directory does not exist
        if exists(self.model_path):
            raise ValueError(f"The model {self.id} already exists.")