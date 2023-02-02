from typing import Union, List
from numpy import ndarray, array
from pandas import DataFrame
from random import seed
from numpy.random import seed as npseed
from tensorflow import random as tf_random
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras.backend import clear_session
from keras import Sequential
from keras.callbacks import EarlyStopping, History
from modules._types import IKerasTrainingConfig, IKerasModelConfig, IKerasModelTrainingHistory,\
    IKerasTrainingConfig, IRegressionTrainingCertificate, IDiscovery, IRegressionTrainingConfig, \
        IRegressionTrainAndTestDatasets, ITestDatasetEvaluation
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.keras_utils.KerasOptimizer import KerasOptimizer, IKerasOptimizerInstance
from modules.keras_utils.KerasLoss import KerasLoss, IKerasLossInstance
from modules.keras_utils.KerasMetric import KerasMetric, IKerasMetricInstance
from modules.keras_utils.KerasModel import KerasModel
from modules.keras_utils.TrainingProgressBar import TrainingProgressBar
from modules.keras_utils.KerasModelSummary import get_summary
from modules.regression.RegressionDiscovery import RegressionDiscovery



class RegressionTraining:
    """RegressionTraining Class

    This class handles the training, evaluating and saving of Keras Regressions.

    Class Properties:
        TRAINING_CONFIG: IKerasTrainingConfig
            The configuration to be used in order to train the models.

    Instance Properties:
        id: str
            A descriptive identifier compatible with filesystems
        description: str
            Important information regarding the model that will be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        learning_rate: float
            The learning rate that will be used to train the model.
        optimizer: IKerasOptimizerInstance
            The optimizer that will be used to train the model.
        loss: IKerasLossInstance
            The loss function that will be used for training.
        metric: IKerasMetricInstance
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
        discovery: RegressionDiscovery
            The instance of the regression discovery.
    """
    # Training Configuration
    TRAINING_CONFIG: IKerasTrainingConfig = {
        "initial_lr": 0.005,
        "decay_steps": 10,
        "decay_rate": 0.1,
        "max_epochs": 1000,
        "patience": 25,
        "batch_size": 512
    }





    ####################
    ## Initialization ##
    ####################



    def __init__(self, config: IRegressionTrainingConfig, datasets: IRegressionTrainAndTestDatasets):
        """Initializes the RegressionTraining Instance.

        Args:
            config: IKerasRegressionTrainingConfig
                The configuration that will be used to train the model.
            datasets: IRegressionTrainAndTestDatasets
                The packed datasets that will be used to train and evaluate the
                regression.

        Raises:
            ValueError:
                If the model is not correctly preffixed.
                If the model's directory already exists.
        """
        # Initialize the id
        if not isinstance(config.get("id"), str) or len(config["id"]) <= 5 or config["id"][0:3] != "KR_":
            raise ValueError(f"The provided regression model id is invalid: {config.get('id')}.")
        self.id: str = config["id"]

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the lookback
        self.lookback: int = config["lookback"]

        # Initialize the predictions output
        self.predictions: int = config["predictions"]

        # Initialize the learning rate
        self.learning_rate: float = config["learning_rate"]

        # Initialize the optimizer function
        self.optimizer: IKerasOptimizerInstance = KerasOptimizer(
            func_name=config["optimizer"], 
            learning_rate=self.learning_rate,
            config=RegressionTraining.TRAINING_CONFIG
        )

        # Initialize the loss function
        self.loss: IKerasLossInstance = KerasLoss(config["loss"])

        # Initialize the metric function
        self.metric: IKerasMetricInstance = KerasMetric(config["metric"])
        
        # Initialize the Keras Model's Configuration and populate the lookback
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["lookback"] = self.lookback
        self.keras_model["predictions"] = self.predictions

        # Make the datasets
        self.train_x, self.train_y, self.test_x, self.test_y = datasets

        # Initialize the Dataset Sizes
        self.train_size: int = self.train_x.shape[0]
        self.test_size: int = self.test_x.shape[0]

        # Initialize the Discovery Instance
        self.discovery: RegressionDiscovery = RegressionDiscovery()

        # Clear the Keras Session
        clear_session()
        seed(Epoch.SEED)
        npseed(Epoch.SEED)
        tf_random.set_seed(Epoch.SEED)









    ##############
    ## Training ##
    ##############





    def train(self) -> IRegressionTrainingCertificate:
        """Compiles, trains and saves the model as well as the training certificate.

        Returns:
            IRegressionTrainingCertificate
        """
        # Store the start time
        start_time: int = Utils.get_time()

        # Initialize the early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            mode="min", 
            patience=RegressionTraining.TRAINING_CONFIG["patience"],
            restore_best_weights=True,
            verbose=0
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
            validation_split=Epoch.VALIDATION_SPLIT, 
            epochs=RegressionTraining.TRAINING_CONFIG["max_epochs"],
            callbacks=[ 
                early_stopping, 
                TrainingProgressBar(RegressionTraining.TRAINING_CONFIG["max_epochs"], "       ") 
            ],
            shuffle=True,
            batch_size=RegressionTraining.TRAINING_CONFIG["batch_size"],
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Predict the test dataset
        print("    4/8) Predicting Test Dataset...")
        preds: List[List[float]] = model.predict(self.test_x, verbose=0).tolist()

        # Evaluate the test dataset
        print("    5/8) Evaluating Test Dataset...")
        test_ds_evaluation: ITestDatasetEvaluation = {
            self.loss.name: float(self.loss(preds, self.test_y)),
            self.metric.name: float(self.metric(preds, self.test_y))
        }

        # Perform the regression discovery
        print("    6/8) Discovering Regression...")
        discovery: IDiscovery = self.discovery.discover(
            features=self.test_x,
            labels=self.test_y,
            preds=preds
        )

        # Build the training certificate
        print("    7/8) Building Certificate...")
        certificate: IRegressionTrainingCertificate = self._build_certificate(
            model=model,
            start_time=start_time,  
            training_history=history, 
            test_ds_evaluation=test_ds_evaluation, 
            discovery=discovery
        )

        # Save the model
        print("    8/8) Saving Model...")
        self._save_model(certificate, model)

        # Return it so it can be added to the batch
        return certificate






    def _build_certificate(
        self,
        model: Sequential,
        start_time: int, 
        training_history: IKerasModelTrainingHistory, 
        test_ds_evaluation: ITestDatasetEvaluation,
        discovery: IDiscovery
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
            test_ds_evaluation: ITestDatasetEvaluation
                The evaluation performed on the test dataset.
            discovery: IDiscovery
                The discovery of the regression.

        Returns:
            IRegressionTrainingCertificate
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
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "metric": self.metric.name,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_ds_evaluation": test_ds_evaluation,

            # Regression Discovery
            "discovery": discovery,

            # The configuration of the Regression
            "regression_config": {
                "id": self.id,
                "description": self.description,
                "lookback": self.lookback,
                "predictions": self.predictions,
                "summary": get_summary(model)
            }
        }






    def _save_model(self, certificate: IRegressionTrainingCertificate, model: Sequential) -> None:
        """Saves a trained model and its certificate.

        Args:
            certificate: IRegressionTrainingCertificate
                The certificate of the model.
            model: Sequential
                The Sequential Instance of the model.
        """
        # Make the directory
        Utils.make_directory(Epoch.PATH.regressions(certificate["id"]))

        # Save the model with the required metadata
        with h5pyFile(Epoch.PATH.regression_model(certificate["id"]), mode="w") as f:
            save_model_to_hdf5(model, f)
            f.attrs["id"] = certificate["id"]
            f.attrs["description"] = certificate["description"]
            f.attrs["lookback"] = certificate["regression_config"]["lookback"]
            f.attrs["predictions"] = certificate["regression_config"]["predictions"]

        # Save the certificate
        Utils.write(Epoch.PATH.regression_certificate(certificate["id"]), certificate)










    #######################
    ## Training Datasets ##
    #######################




    @staticmethod
    def make_train_and_test_datasets() -> IRegressionTrainAndTestDatasets:
        """Builds a tuple containing the features and labels for the train and test datasets.

        Returns:
            IRegressionTrainAndTestDatasets
            (train_x, train_y, test_x, test_y)
        """
        # Init the df, grabbing only the close prices
        df: DataFrame = Candlestick.NORMALIZED_PREDICTION_DF[["c"]].copy()

        # Init the number of rows and the split that will be applied
        rows: int = df.shape[0]
        split: int = int(rows * Epoch.TRAIN_SPLIT)

        # Init raw features and labels
        features_raw: Union[List[List[float]], ndarray] = []
        labels_raw: Union[List[List[float]], ndarray] = []

        # Iterate over the normalized ds and build the features & labels
        for i in range(Epoch.REGRESSION_LOOKBACK, rows):
            # Make sure there are enough items remaining
            if i < (rows-Epoch.REGRESSION_PREDICTIONS):
                features_raw.append(df.iloc[i-Epoch.REGRESSION_LOOKBACK:i, 0])
                labels_raw.append(df.iloc[i:i+Epoch.REGRESSION_PREDICTIONS, 0])

        # Convert the features and labels into np arrays
        features = array(features_raw)
        labels = array(labels_raw)

        # Finally, return the split datasets
        return features[:split], labels[:split], features[split:], labels[split:]












    ############################
    ## Certificate Management ##
    ############################




    @staticmethod
    def get_certificate(id: str) -> Union[IRegressionTrainingCertificate, None]:
        """Retrieves a trained model's certificate. If it doesnt exist it 
        returns None.

        Args:
            id: str
                The identifier of the model.

        Returns:
            Union[IRegressionTrainingCertificate, None]
        """
        return Utils.read(Epoch.PATH.regression_certificate(id), allow_empty=True)




    



    @staticmethod
    def save_certificates_batch(batch_name: str, certificates: List[IRegressionTrainingCertificate]) -> None:
        """Retrieves a trained model's certificate. If it doesnt exist it 
        returns None.

        Args:
            batch_name: str
                The name of the batch that contains the certificates that are about
                to be saved.
            certificates: List[IRegressionTrainingCertificate]
                The list of training certificates to be saved in a single file.
        """
        # Init the path of the batch
        batch_path: str = f"{Epoch.PATH.regression_batched_certificates()}/{batch_name}.json"

        # Save the list of certificates
        Utils.write(batch_path, certificates)