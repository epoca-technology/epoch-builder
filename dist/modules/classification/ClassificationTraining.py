from typing import Union, List, Tuple
from os import makedirs
from os.path import exists
from pandas import DataFrame, concat
from json import dumps
from h5py import File as h5pyFile
from tensorflow import random as tf_random
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, History
from modules.types import IKerasTrainingTypeConfig, IKerasModelConfig, IKerasModelTrainingHistory, IModel,\
    ITrainingDataFile, ICompressedTrainingData, IClassificationTrainingConfig, ITrainingDataSummary, \
        IClassificationTrainingCertificate, IModelEvaluation
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.keras_models.KerasPath import KERAS_PATH
from modules.keras_models.KerasModel import KerasModel
from modules.keras_models.LearningRateSchedule import LearningRateSchedule
from modules.keras_models.KerasModelSummary import get_summary
from modules.classification.TrainingDataCompression import decompress_training_data
from modules.model_evaluation.ModelEvaluation import evaluate
        




class ClassificationTraining:
    """ClassificationTraining Class

    This class handles the training of a Classification Model.

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
            A descriptive identifier compatible with filesystems.
        model_path: str
            The directory in which the model will be stored.
        description: str
            Important information regarding the model that will be trained.
        models: List[IModel]
            The list of ArimaModel|RegressionModel used to generate the training data.
        optimizer: Union[adam.Adam, rmsprop.RMSProp]                    "adam"|"rmsprop"
            The optimizer that will be used to train the model.
        loss: Union[CategoricalCrossentropy, BinaryCrossentropy]        "categorical_crossentropy"|"binary_crossentropy"
            The loss function that will be used for training.
        metric: Union[CategoricalAccuracy, BinaryAccuracy]              "categorical_accuracy"|"binary_accuracy" 
            The metric function that will be used for training.
        keras_model: IKerasModelConfig
            The configuration that will be used to build the Keras Model.
        train_x: DataFrame
            The train features df
        train_y: DataFrame
            The train labels df
        test_x: DataFrame
            The test features df
        test_y: DataFrame
            The test labels df
        training_data_summary: ITrainingDataSummary
            The summary of the training data that will be attached to the training certificate
    """
    # Hyperparams Training Configuration
    HYPERPARAMS_TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "train_split": 0.8,
        "initial_lr": 0.01,
        "decay_steps": 1.5,
        "decay_rate": 0.28,
        "epochs": 100,
        "patience": 30
    }

    # Shortlisted Training Configuration
    SHORTLISTED_TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "train_split": 0.8,
        "initial_lr": 0.01,
        "decay_steps": 2,
        "decay_rate": 0.065,
        "epochs": 500,
        "patience": 100
    }



    ## Initialization ##



    def __init__(
        self, 
        training_data_file: ITrainingDataFile, 
        config: IClassificationTrainingConfig, 
        hyperparams_mode: bool=False,
        test_mode: bool = False
    ):
        """Initializes the ClassificationTraining Instance.

        Args:
            training_data_file: ITrainingDataFile
                The training data file that will be used to train and evaluate the model.
            config: IClassificationTrainingConfig
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
            ClassificationTraining.HYPERPARAMS_TRAINING_CONFIG if self.hyperparams_mode \
                else ClassificationTraining.SHORTLISTED_TRAINING_CONFIG
        
        # Initialize the id
        self.id: str = config["id"]
        if self.id[0:2] != "C_":
            raise ValueError("The ID of the ClassificationModel must be preffixed with C_")

        # Initialize the Model's path
        self.model_path: str = f"{KERAS_PATH['models']}/{self.id}"

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the models data as well as the regression instances
        self.models: List[IModel] = training_data_file["models"]

        # Initialize the optimizer function
        self.optimizer: Union[adam.Adam, rmsprop.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[CategoricalCrossentropy, BinaryCrossentropy] = self._get_loss(config["loss"])

        # Initialize the metric function
        self.metric: Union[CategoricalAccuracy, BinaryAccuracy] = self._get_metric(config["metric"])

        # Initialize the Training Data
        train_x, train_y, test_x, test_y = self._make_datasets(training_data_file["training_data"])
        self.train_x: DataFrame = train_x
        self.train_y: DataFrame = train_y
        self.test_x: DataFrame = test_x
        self.test_y: DataFrame = test_y

        # Initialize the Training Data Summary
        self.training_data_summary: ITrainingDataSummary = self._get_training_data_summary(training_data_file)

        # Initialize the Keras Model's Configuration
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["features_num"] = self.training_data_summary["features_num"]

        # Initialize the model's directory if not unit testing
        if not self.test_mode:
            self._init_model_dir()









    def _get_optimizer(self, func_name: str) -> Union[adam.Adam, rmsprop.RMSProp]:
        """Based on a optimizer function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the optimizer function to be used.

        Returns:
            Union[adam.Adam, rmsprop.RMSProp]

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
        if func_name == "adam":
            return adam.Adam(lr_schedule)
        elif func_name == "rmsprop":
            return rmsprop.RMSProp(lr_schedule)
        else:
            raise ValueError(f"The optimizer function for {func_name} was not found.")






    def _get_loss(self, func_name: str) -> Union[CategoricalCrossentropy, BinaryCrossentropy]:
        """Based on a loss function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the loss function to be used.

        Returns:
            Union[CategoricalCrossentropy, BinaryCrossentropy]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == "categorical_crossentropy":
            return CategoricalCrossentropy()
        elif func_name == "binary_crossentropy":
            return BinaryCrossentropy()
        else:
            raise ValueError(f"The loss function for {func_name} was not found.")







    def _get_metric(self, func_name: str) -> Union[CategoricalAccuracy, BinaryAccuracy]:
        """Based on a metric function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the loss function to be used.

        Returns:
            Union[CategoricalAccuracy, BinaryAccuracy]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == "categorical_accuracy":
            return CategoricalAccuracy()
        elif func_name == "binary_accuracy":
            return BinaryAccuracy()
        else:
            raise ValueError(f"The metric function for {func_name} was not found.")











    def _make_datasets(self, training_data: ICompressedTrainingData) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """Splits the Classification Training Data into train and test dataframes.

        Args:
            training_data: ICompressedTrainingData
                The Training Data to be decompressed and split.

        Returns:
            Tuple[DataFrame, DataFrame, DataFrame, DataFrame] 
            (train_x, train_y, test_x, test_y)
        """
        # Decompress the training data
        df: DataFrame = decompress_training_data(training_data)
        
        # Initialize the total rows and the split size
        rows: int = df.shape[0]
        split: int = int(rows*self.training_config["train_split"])

        # Initialize the features dfs
        train_x: DataFrame = df[:split]
        test_x: DataFrame = df[split:]

        # Initialize the labels dfs
        train_y: DataFrame = concat([train_x.pop(x) for x in ["up", "down"]], axis=1)
        test_y: DataFrame = concat([test_x.pop(x) for x in ["up", "down"]], axis=1)

        # Return the packed dfs
        return train_x, train_y, test_x, test_y







    def _get_training_data_summary(self, file: ITrainingDataFile) -> ITrainingDataSummary:
        """Returns a brief overview of a Training Data File.

        Args:
            file: ITrainingDataFile
                The file generated by the training data execution.

        Returns:
            ITrainingDataSummary
        """
        return {
            "regression_selection_id": file["regression_selection_id"],
            "id": file["id"],
            "description": file["description"],
            "start": file["start"],
            "end": file["end"],
            "train_size": self.train_x.shape[0],
            "test_size": self.test_x.shape[0],
            "steps": file["steps"],
            "up_percent_change": file["up_percent_change"],
            "down_percent_change": file["down_percent_change"],
            "include_rsi": file["include_rsi"],
            "include_stoch": file["include_stoch"],
            "include_aroon": file["include_aroon"],
            "include_stc": file["include_stc"],
            "include_mfi": file["include_mfi"],
            "features_num": file["features_num"]
        }







    




    ## Training ##






    def train(self) -> IClassificationTrainingCertificate:
        """Compiles, trains and saves the model as well as the training certificate.

        Returns:
            IClassificationTrainingCertificate
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_categorical_accuracy" if self.metric.name == "categorical_accuracy" else "val_binary_accuracy", 
            mode="max", 
            min_delta=0.001, 
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
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])
  
        # Train the model
        if not self.hyperparams_mode:
            print("    3/7) Training Model...")
        history_object: History = model.fit(
            self.train_x,
            self.train_y,
            validation_split=0.2,
            epochs=self.training_config["epochs"],
            shuffle=True,
            callbacks=[ early_stopping ],
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        if not self.hyperparams_mode:
            print("    4/7) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.test_x, self.test_y, verbose=0) # [loss, accuracy]

        # Save the model
        if not self.hyperparams_mode:
            print("    5/7) Saving Model...")
        self._save_model(model)

        # Perform the Classification Evaluation
        classification_evaluation: IModelEvaluation = self._evaluate_classification()

        # Save the certificate
        if not self.hyperparams_mode:
            print("    7/7) Saving Certificate...")
        certificate: IClassificationTrainingCertificate = self._save_certificate(
            start_time, 
            model, 
            history, 
            test_evaluation,
            classification_evaluation
        )

        # Return it so it can be added to the batch
        return certificate












    def _evaluate_classification(self) -> IModelEvaluation:
        """Loads the trained model that has just been saved and performs a runs
        an evaluation that is similar to the backtest. This eval will only run
        on the test dataset

        Returns:
            IModelEvaluation
        """
        # Init the number of rows and the split that will be applied
        rows: int = Candlestick.PREDICTION_DF.shape[0]
        split: int = int(rows * self.training_config["train_split"])

        # Initialize the first open time of the test dataset
        first_ot: int = Candlestick.PREDICTION_DF[split:split+1].iloc[0]["ot"]

        # Finally, evaluate the model
        return evaluate(
            model_config={
                "id": self.id,
                "classification_models": [{ "classification_id": self.id, "interpreter": { "min_probability": 0.60 }}]
            },
            start_timestamp=first_ot,
            price_change_requirement=self.training_data_summary["up_percent_change"],
            hyperparams_mode=self.hyperparams_mode
        )
















    ## Trained Model Saving ##





    def _save_model(self, model: Sequential) -> None:
        """Saves a trained model in the output directory.

        Args:
            model: Sequential
                The instance of the trained model.
        """
        # Create the model's directory
        makedirs(self.model_path)
        
        # Save the model with the required metadata
        with h5pyFile(f"{self.model_path}/model.h5", mode="w") as f:
            save_model_to_hdf5(model, f)
            f.attrs["id"] = self.id
            f.attrs["description"] = self.description
            f.attrs["training_data_id"] = self.training_data_summary["id"]
            f.attrs["models"] = dumps(self.models)
            f.attrs["include_rsi"] = self.training_data_summary["include_rsi"]
            f.attrs["include_stoch"] = self.training_data_summary["include_stoch"]
            f.attrs["include_aroon"] = self.training_data_summary["include_aroon"]
            f.attrs["include_stc"] = self.training_data_summary["include_stc"]
            f.attrs["include_mfi"] = self.training_data_summary["include_mfi"]
            f.attrs["features_num"] = self.training_data_summary["features_num"]







    def _save_certificate(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        classification_evaluation: IModelEvaluation
    ) -> IClassificationTrainingCertificate:
        """Saves the training certificate in the same directory as the model.

        Args:
            start_time: int
                The time in which the training started.
            model: Sequential
                The instance of the trained model.
            training_history: IKerasModelTrainingHistory
                The dictionary containing the training history.
            test_evaluation: List[float]
                The results when evaluating the test dataset.
            classification_evaluation: IModelEvaluation
                The results of the classification evaluation.

        Returns:
            IClassificationTrainingCertificate
        """
        # Build the certificate
        certificate: IClassificationTrainingCertificate = self._get_certificate(
            model, 
            start_time, 
            training_history, 
            test_evaluation,
            classification_evaluation
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
        classification_evaluation: IModelEvaluation
    ) -> IClassificationTrainingCertificate:
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
            classification_evaluation: IModelEvaluation
                The results of the classification evaluation.

        Returns:
            IRegressionTrainingCertificate
        """
        return {
            # Identification
            "id": self.id,
            "description": self.description,

            # Training Data
            "training_data_summary": self.training_data_summary,

            # Training Configuration
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "metric": self.metric.name,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_evaluation": test_evaluation,

            # Post Training Evaluation
            "classification_evaluation": classification_evaluation,

            # The configuration of the Classification
            "classification_config": {
                "id": self.id,
                "description": self.description,
                "training_data_id": self.training_data_summary["id"],
                "models": self.models,
                "include_rsi": self.training_data_summary["include_rsi"],
                "include_stoch": self.training_data_summary["include_stoch"],
                "include_aroon": self.training_data_summary["include_aroon"],
                "include_stc": self.training_data_summary["include_stc"],
                "include_mfi": self.training_data_summary["include_mfi"],
                "features_num": self.training_data_summary["features_num"],
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