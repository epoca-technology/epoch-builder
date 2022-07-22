from typing import Union, List, Tuple
from pandas import DataFrame, concat
from json import dumps
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, History
from modules.types import IKerasTrainingTypeConfig, IKerasModelConfig, IKerasModelTrainingHistory, IModel,\
    ITrainingDataFile, ICompressedTrainingData, IClassificationTrainingConfig, ITrainingDataSummary, \
        IClassificationTrainingCertificate, IModelEvaluation, IKerasOptimizer, IKerasClassificationLoss,\
            IKerasClassificationMetric, ITrainableModelType
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.keras_models.KerasModel import KerasModel
from modules.keras_models.LearningRateSchedule import LearningRateSchedule
from modules.keras_models.KerasModelSummary import get_summary
from modules.classification_training_data.TrainingDataCompression import decompress_training_data
from modules.model_evaluation.ModelEvaluation import evaluate
        




class ClassificationTraining:
    """ClassificationTraining Class

    This class handles the training of a Classification Model.

    Class Properties:
        TRAINING_CONFIG: IKerasTrainingTypeConfig
            The configuration to be used in order to train the models

    Instance Properties:
        test_mode: bool
            If running from unit tests, it won't check the model's directory.
        model_type: ITrainableModelType
            The type of model that will be trained.
        id: str
            A descriptive identifier compatible with filesystems.
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
    # Training Configuration
    TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "train_split": 0.9,
        "initial_lr": 0.01,
        "decay_steps": 1,
        "decay_rate": 0.35,
        "epochs": 100,
        "patience": 40,
        "batch_size": 32
    }



    ## Initialization ##



    def __init__(
        self, 
        training_data_file: ITrainingDataFile, 
        config: IClassificationTrainingConfig, 
        datasets: Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], None]=None,
        test_mode: bool = False
    ):
        """Initializes the ClassificationTraining Instance.

        Args:
            training_data_file: ITrainingDataFile
                The training data file that will be used to train and evaluate the model.
            config: IClassificationTrainingConfig
                The configuration that will be used to train the model.
            datasets: Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], None]
                The datasets that will be used to train and test the model. This data
                can be built in the CLI script or in the instance itself.
            test_mode: bool
                If running from unit tests, it won't check the model's directory.

        Raises:
            ValueError:
                If the model is not correctly preffixed.
            RuntimeError:
                If the model's directory already exists.
        """
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Init the type of model
        self.model_type: ITrainableModelType = "keras_classification"
        
        # Initialize the id
        self.id: str = config["id"]
        if self.id[0:2] != "C_":
            raise ValueError("The ID of the ClassificationModel must be preffixed with C_")

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

        # Make the datasets if they weren't provided
        if datasets is None:
            self.train_x, self.train_y, self.test_x, self.test_y = ClassificationTraining.make_datasets(
                training_data=training_data_file["training_data"]
            )

        # Otherwise, unpack the provided datasets
        else:
            self.train_x, self.train_y, self.test_x, self.test_y = datasets

        # Initialize the Training Data Summary
        self.training_data_summary: ITrainingDataSummary = self._get_training_data_summary(training_data_file)

        # Initialize the Keras Model's Configuration
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["features_num"] = self.training_data_summary["features_num"]

        # Make sure the model does not exist if not unit testing
        if not self.test_mode and Epoch.FILE.model_exists(self.id, self.model_type):
            raise RuntimeError(f"Cannot train the model {self.id} because it already exists.")









    def _get_optimizer(self, func_name: IKerasOptimizer) -> Union[adam.Adam, rmsprop.RMSProp]:
        """Based on a optimizer function name, it will return the instance ready to be initialized.

        Args:
            func_name: IKerasOptimizer
                The name of the optimizer function to be used.

        Returns:
            Union[adam.Adam, rmsprop.RMSProp]

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        # Initialize the Learning Rate Schedule
        lr_schedule: InverseTimeDecay = LearningRateSchedule(
            initial_learning_rate=ClassificationTraining.TRAINING_CONFIG["initial_lr"],
            decay_steps=ClassificationTraining.TRAINING_CONFIG["decay_steps"],
            decay_rate=ClassificationTraining.TRAINING_CONFIG["decay_rate"]
        )

        # Return the Optimizer Instance
        if func_name == "adam":
            return adam.Adam(lr_schedule)
        elif func_name == "rmsprop":
            return rmsprop.RMSProp(lr_schedule)
        else:
            raise ValueError(f"The optimizer function for {func_name} was not found.")






    def _get_loss(self, func_name: IKerasClassificationLoss) -> Union[CategoricalCrossentropy, BinaryCrossentropy]:
        """Based on a loss function name, it will return the instance ready to be initialized.

        Args:
            func_name: IKerasClassificationLoss
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







    def _get_metric(self, func_name: IKerasClassificationMetric) -> Union[CategoricalAccuracy, BinaryAccuracy]:
        """Based on a metric function name, it will return the instance ready to be initialized.

        Args:
            func_name: IKerasClassificationMetric
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











    @staticmethod
    def make_datasets(training_data: ICompressedTrainingData) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
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
        split: int = int(rows * ClassificationTraining.TRAINING_CONFIG["train_split"])

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
            patience=ClassificationTraining.TRAINING_CONFIG["patience"],
            #restore_best_weights=True
        )

        # Retrieve the Keras Model
        print("    1/7) Initializing Model...")
        model: Sequential = KerasModel(config=self.keras_model)

        # Compile the model
        print("    2/7) Compiling Model...")
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])
  
        # Train the model
        print("    3/7) Training Model...")
        history_object: History = model.fit(
            self.train_x,
            self.train_y,
            validation_split=0.2,
            epochs=ClassificationTraining.TRAINING_CONFIG["epochs"],
            shuffle=True,
            callbacks=[ early_stopping ],
            batch_size=ClassificationTraining.TRAINING_CONFIG["batch_size"],
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        print("    4/7) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.test_x, self.test_y, verbose=0) # [loss, accuracy]

        # Save the model
        print("    5/7) Saving Model...")
        self._save_model(model)

        # Perform the Classification Evaluation
        classification_evaluation: IModelEvaluation = evaluate(
            model_config={
                "id": self.id,
                "classification_models": [{ "classification_id": self.id, "interpreter": { "min_probability": 0.6 }}]
            },
            price_change_requirement=self.training_data_summary["up_percent_change"],
            progress_bar_description="    6/7) Evaluating ClassificationModel"
        )

        # Save the certificate
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















    ## Trained Model Saving ##





    def _save_model(self, model: Sequential) -> None:
        """Saves a trained model in the output directory.

        Args:
            model: Sequential
                The instance of the trained model.
        """
        # Create the model's directory
        Epoch.FILE.make_active_model_dir(self.id)
        
        # Save the model with the required metadata
        with h5pyFile(Epoch.FILE.get_active_model_path(self.id, self.model_type), mode="w") as f:
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
        certificate: IClassificationTrainingCertificate = self._build_certificate(
            model, 
            start_time, 
            training_history, 
            test_evaluation,
            classification_evaluation
        )

        # Save the file
        Epoch.FILE.save_training_certificate(certificate)

        # Finally, return it so it can be added to the batch
        return certificate







    def _build_certificate(
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