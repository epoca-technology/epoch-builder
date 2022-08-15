from typing import Union, List
from json import dumps
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, History
from modules._types import IKerasTrainingTypeConfig, IKerasModelConfig, IKerasModelTrainingHistory, IModel,\
    ITrainingDataFile, IKerasClassificationTrainingConfig, ITrainingDataSummary, IKerasClassificationTrainingCertificate,\
         IModelEvaluation, IKerasOptimizer, IKerasClassificationLoss, IKerasClassificationMetric, ITrainableModelType,\
         IClassificationDatasets, IDiscovery, IDiscoveryPayload
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.model.ModelType import validate_id
from modules.keras_models.KerasModel import KerasModel
from modules.keras_models.LearningRateSchedule import LearningRateSchedule
from modules.keras_models.KerasTrainingProgress import KerasTrainingProgressBar, training_passed
from modules.keras_models.KerasModelSummary import get_summary
from modules.classification_training_data.ClassificationTrainingData import ClassificationTrainingData
from modules.keras_classification.KerasClassification import KerasClassification
from modules.discovery.ClassificationDiscovery import discover
from modules.model.KerasClassificationModel import KerasClassificationModel
from modules.model_evaluation.ModelEvaluation import evaluate




class KerasClassificationTraining:
    """KerasClassificationTraining Class

    This class handles the training of a Keras Classification Model.

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
        regressions: List[IModel]
            The list of ArimaModel|RegressionModel used to generate the training data.
        learning_rate: float
            The learning rate that will be used to train the model.
        optimizer: Union[adam.Adam, rmsprop.RMSProp]                    "adam"|"rmsprop"
            The optimizer that will be used to train the model.
        loss: Union[CategoricalCrossentropy, BinaryCrossentropy]        "categorical_crossentropy"|"binary_crossentropy"
            The loss function that will be used for training.
        metric: Union[CategoricalAccuracy, BinaryAccuracy]              "categorical_accuracy"|"binary_accuracy" 
            The metric function that will be used for training.
        batch_size: int
            Number of samples per gradient update. Can be adjusted based on the network that will be trained.
        keras_model: IKerasModelConfig
            The configuration that will be used to build the Keras Model.
        train_x: ndarray
        train_y: ndarray
        test_x: ndarray
        test_y: ndarray
            Features and labels. The training data.
        training_data_summary: ITrainingDataSummary
            The summary of the training data that will be attached to the training certificate
    """
    # Training Configuration
    TRAINING_CONFIG: IKerasTrainingTypeConfig = {
        "initial_lr": 0.01,
        "decay_steps": 1,
        "decay_rate": 0.35,
        "epochs": 100,
        "patience": 15,
        "batch_size": 32
    }





    ## Initialization ##



    def __init__(
        self, 
        training_data_file: ITrainingDataFile, 
        config: IKerasClassificationTrainingConfig, 
        datasets: IClassificationDatasets,
        test_mode: bool = False
    ):
        """Initializes the KerasClassificationTraining Instance.

        Args:
            training_data_file: ITrainingDataFile
                The training data file that will be used to train and evaluate the model.
            config: IKerasClassificationTrainingConfig
                The configuration that will be used to train the model.
            datasets: IClassificationDatasets
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
        validate_id("KerasClassificationModel", config["id"])
        self.id: str = config["id"]

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the regressions data as well as the regression instances
        self.regressions: List[IModel] = training_data_file["regressions"]

        # Initialize the learning rate
        self.learning_rate: float = config["learning_rate"]

        # Initialize the optimizer function
        self.optimizer: Union[adam.Adam, rmsprop.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[CategoricalCrossentropy, BinaryCrossentropy] = self._get_loss(config["loss"])

        # Initialize the metric function
        self.metric: Union[CategoricalAccuracy, BinaryAccuracy] = self._get_metric(config["metric"])

        # Initialize the batch size
        self.batch_size: int = self._get_batch_size()

        # Unpack the provided datasets
        self.train_x, self.train_y, self.test_x, self.test_y = datasets

        # Initialize the Training Data Summary
        self.training_data_summary: ITrainingDataSummary = ClassificationTrainingData.get_training_data_summary(
            file=training_data_file,
            train_size=self.train_x.shape[0],
            test_size=self.test_x.shape[0]
        )

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
        lr_schedule: Union[InverseTimeDecay, float] = LearningRateSchedule(
            learning_rate=self.learning_rate,
            initial_learning_rate=KerasClassificationTraining.TRAINING_CONFIG["initial_lr"],
            decay_steps=KerasClassificationTraining.TRAINING_CONFIG["decay_steps"],
            decay_rate=KerasClassificationTraining.TRAINING_CONFIG["decay_rate"]
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







    def _get_batch_size(self) -> int:
        """Retrieves the batch size that will be used to train the models based
        on the network.

        Returns:
            int
        """
        #if "DNN" in self.id:
        #    return 32
        #elif "CNN" in self.id:
        #    return 64
        #elif "LSTM" in self.id:
        #    return 128
        #elif "CLSTM" in self.id:
        #    return 256
        #else:
        #    return KerasClassificationTraining.TRAINING_CONFIG["batch_size"]
        return KerasClassificationTraining.TRAINING_CONFIG["batch_size"]
            









    




    ## Training ##






    def train(self) -> IKerasClassificationTrainingCertificate:
        """Compiles, trains and saves the model as well as the training certificate.

        Returns:
            IKerasClassificationTrainingCertificate
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_categorical_accuracy" if self.metric.name == "categorical_accuracy" else "val_binary_accuracy", 
            mode="max", 
            min_delta=0.001, 
            patience=KerasClassificationTraining.TRAINING_CONFIG["patience"],
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
            epochs=KerasClassificationTraining.TRAINING_CONFIG["epochs"],
            shuffle=True,
            callbacks=[ 
                early_stopping, 
                KerasTrainingProgressBar(KerasClassificationTraining.TRAINING_CONFIG["epochs"], "       ") 
            ],
            batch_size=self.batch_size,
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        print("    4/8) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.test_x, self.test_y, verbose=0) # [loss, accuracy]

        # Perform the classification discovery
        print("    5/8) Discovering KerasClassification")
        discovery, discovery_payload = discover(
            classification=KerasClassification(self.id, {
                "model": model,
                "training_data_id": self.training_data_summary["id"],
                "include_rsi": self.training_data_summary["include_rsi"],
                "include_aroon": self.training_data_summary["include_aroon"],
                "features_num": self.training_data_summary["features_num"],
                "regressions": self.regressions,
                "price_change_requirement": self.training_data_summary["price_change_requirement"]
            }),
            progress_bar_description="       ",
            training_passed=training_passed(history, KerasClassificationTraining.TRAINING_CONFIG["epochs"])
        )

        # Save the model
        print("    6/8) Saving Model...")
        self._save_model(model, discovery)

        # Perform the Classification Evaluation
        print("    7/8) Evaluating KerasClassificationModel")
        classification_model: KerasClassificationModel = KerasClassificationModel(
            config={
                "id": self.id,
                "keras_classifications": [ {"classification_id": self.id } ]
            },
            enable_cache=False
        )
        classification_evaluation: IModelEvaluation = evaluate(
            model=classification_model,
            price_change_requirement=self.training_data_summary["price_change_requirement"],
            progress_bar_description="       ",
            discovery_completed=discovery_payload["early_stopping"] == None
        )

        # Save the certificate
        print("    8/8) Saving Certificate...")
        certificate: IKerasClassificationTrainingCertificate = self._save_certificate(
            start_time, 
            model, 
            history, 
            test_evaluation,
            discovery,
            discovery_payload,
            classification_evaluation
        )

        # Return it so it can be added to the batch
        return certificate















    ## Trained Model Saving ##





    def _save_model(self, model: Sequential, discovery: IDiscovery) -> None:
        """Saves a trained model in the output directory.

        Args:
            model: Sequential
                The instance of the trained model.
            discovery: IDiscovery
                The result of the classification discovery.
        """
        # Create the model's directory
        Epoch.FILE.make_active_model_dir(self.id)
        
        # Save the model with the required metadata
        with h5pyFile(Epoch.FILE.get_active_model_path(self.id, self.model_type), mode="w") as f:
            save_model_to_hdf5(model, f)
            f.attrs["id"] = self.id
            f.attrs["description"] = self.description
            f.attrs["training_data_id"] = self.training_data_summary["id"]
            f.attrs["regressions"] = dumps(self.regressions)
            f.attrs["include_rsi"] = self.training_data_summary["include_rsi"]
            f.attrs["include_aroon"] = self.training_data_summary["include_aroon"]
            f.attrs["features_num"] = self.training_data_summary["features_num"]
            f.attrs["price_change_requirement"] = self.training_data_summary["price_change_requirement"]
            f.attrs["discovery"] = dumps(discovery)







    def _save_certificate(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        discovery: IDiscovery,
        discovery_payload: IDiscoveryPayload,
        classification_evaluation: IModelEvaluation
    ) -> IKerasClassificationTrainingCertificate:
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
            discovery: IDiscovery
            discovery_payload: IDiscoveryPayload
                The discovery and the payload of the regression.
            classification_evaluation: IModelEvaluation
                The results of the classification evaluation.

        Returns:
            IKerasClassificationTrainingCertificate
        """
        # Build the certificate
        certificate: IKerasClassificationTrainingCertificate = self._build_certificate(
            model, 
            start_time, 
            training_history, 
            test_evaluation,
            discovery,
            discovery_payload,
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
        discovery: IDiscovery,
        discovery_payload: IDiscoveryPayload,
        classification_evaluation: IModelEvaluation
    ) -> IKerasClassificationTrainingCertificate:
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
            classification_evaluation: IModelEvaluation
                The results of the classification evaluation.

        Returns:
            IKerasClassificationTrainingCertificate
        """
        return {
            # Identification
            "id": self.id,
            "description": self.description,

            # Training Data
            "training_data_summary": self.training_data_summary,

            # Training Configuration
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

            # Classification Discovery
            "discovery": discovery_payload,

            # Post Training Evaluation
            "classification_evaluation": classification_evaluation,

            # The configuration of the Classification
            "classification_config": {
                "id": self.id,
                "description": self.description,
                "training_data_id": self.training_data_summary["id"],
                "regressions": self.regressions,
                "include_rsi": self.training_data_summary["include_rsi"],
                "include_aroon": self.training_data_summary["include_aroon"],
                "features_num": self.training_data_summary["features_num"],
                "discovery": discovery,
                "summary": get_summary(model)
            }
        }