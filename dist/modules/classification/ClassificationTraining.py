from typing import Union, List, Tuple
from os import makedirs
from os.path import exists
from pandas import DataFrame, concat
from json import dumps
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2, rmsprop_v2
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy as CategoricalAccuracyMetric
from keras.callbacks import EarlyStopping, History
from h5py import File as h5pyFile
from modules.utils import Utils
from modules.model import IModel
from modules.keras_models import KerasModel, IKerasModelConfig, IKerasModelTrainingHistory, get_summary, KERAS_PATH
from modules.classification import ITrainingDataFile, ICompressedTrainingData, decompress_training_data, \
    IClassificationTrainingConfig, ITrainingDataSummary, IClassificationTrainingCertificate




class ClassificationTraining:
    """ClassificationTraining Class

    This class handles the training of a Classification Model.

    Class Properties:
        TRAIN_SPLIT: float
            The split that will be used on the complete df in order to generate train and test 
                features & labels dfs
        MAX_EPOCHS: int
            The maximum amount of epochs the training process can go through.
        EARLY_STOPPING_PATIENCE: int
            The number of epochs it will allow to be executed without showing a performance.

    Instance Properties:
        id: str
            A descriptive identifier compatible with filesystems.
        model_path: str
            The directory in which the model will be stored.
        description: str
            Important information regarding the model that will be trained.
        models: List[IModel]
            The list of ArimaModel|RegressionModel used to generate the training data.
        learning_rate: float
            The learning rate to be used by the optimizer. If None is provided it uses the default
            one.   
        optimizer: Union[adam_v2.Adam, rmsprop_v2.RMSProp]
            The optimizer that will be used to train the model.
        loss: CategoricalCrossentropy
            The loss function that will be used for training.
        metric: CategoricalAccuracyMetric
            The metric function that will be used for training.
        batch_size: int
            The size of the training dataset batches.
        shuffle_data: bool
            If True, it will shuffle the train, val and test datasets prior to training.
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


    # Train and Test DataFrame Split
    TRAIN_SPLIT: float = 0.8

    # The maximum number of EPOCHs a model can go through during training
    MAX_EPOCHS: int = 1000

    # The max number of training epochs that can occur without showing improvements.
    EARLY_STOPPING_PATIENCE: int = 15





    ## Initialization ##



    def __init__(self, training_data_file: ITrainingDataFile, config: IClassificationTrainingConfig):
        """Initializes the ClassificationTraining Instance.

        Args:
            training_data_file: ITrainingDataFile
                The training data file that will be used to train and evaluate the model.
            config: IClassificationTrainingConfig
                The configuration that will be used to train the model.

        Raises:
            ValueError:
                If the model is not correctly preffixed.
                If the model's directory already exists.
        """
        # Initialize the id
        self.id: str = config['id']
        if self.id[0:2] != 'C_':
            raise ValueError("The ID of the ClassificationModel must be preffixed with C_")

        # Initialize the Model's path
        self.model_path: str = f"{KERAS_PATH['models']}/{self.id}"

        # Initialize the description
        self.description: str = config['description']

        # Initialize the models
        self.models: List[IModel] = training_data_file['models']

        # Initialize the Learning Rate
        self.learning_rate: float = config['learning_rate']

        # Initialize the optimizer function
        self.optimizer: Union[adam_v2.Adam, rmsprop_v2.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: CategoricalCrossentropy = self._get_loss(config['loss'])

        # Initialize the metric function
        self.metric: CategoricalAccuracyMetric = self._get_metric(config['metric'])

        # Initialize the Batch Size
        self.batch_size: int = config["batch_size"]

        # Initialize the Data Shuffling
        self.shuffle_data: bool = config["shuffle_data"]

        # Initialize the Keras Model's Configuration
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["features_num"] = len(self.models)

        # Initialize the Training Data
        train_x, train_y, test_x, test_y = self._get_data(training_data_file['training_data'])
        self.train_x: DataFrame = train_x
        self.train_y: DataFrame = train_y
        self.test_x: DataFrame = test_x
        self.test_y: DataFrame = test_y

        # Initialize the Training Data Summary
        self.training_data_summary: ITrainingDataSummary = self._get_training_data_summary(training_data_file)

        # Initialize the model's directory
        self._init_model_dir()








    def _get_data(self, training_data: ICompressedTrainingData) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
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
        
        # Initialize the total rows
        rows: int = df.shape[0]

        # Initialize the features dfs
        train_x: DataFrame = df[:int(rows*ClassificationTraining.TRAIN_SPLIT)]
        test_x: DataFrame = df[int(rows*ClassificationTraining.TRAIN_SPLIT):]

        # Initialize the labels dfs
        train_y: DataFrame = concat([train_x.pop(x) for x in ['up', 'down']], axis=1)
        test_y: DataFrame = concat([test_x.pop(x) for x in ['up', 'down']], axis=1)

        # Return the packed dfs
        return train_x, train_y, test_x, test_y






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






    def _get_loss(self, func_name: str) -> CategoricalCrossentropy:
        """Based on a loss function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the loss function to be used.

        Returns:
            CategoricalCrossentropy

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == 'cc':
            return CategoricalCrossentropy()
        else:
            raise ValueError(f"The loss function for {func_name} was not found.")







    def _get_metric(self, func_name: str) -> CategoricalAccuracyMetric:
        """Based on a metric function name, it will return the instance ready to be initialized.

        Args:
            func_name: str
                The name of the loss function to be used.

        Returns:
            CategoricalAccuracyMetric

        Raises:
            ValueError:
                If the function name does not match any function in the conditionings.
        """
        if func_name == 'ca':
            return CategoricalAccuracyMetric()
        else:
            raise ValueError(f"The metric function for {func_name} was not found.")







    def _get_training_data_summary(self, file: ITrainingDataFile) -> ITrainingDataSummary:
        """Returns a brief overview of a Training Data File.

        Args:
            file: ITrainingDataFile
                The file generated by the training data execution.

        Returns:
            ITrainingDataSummary
        """
        return {
            "id": file['id'],
            "description": file['description'],
            "start": file['start'],
            "end": file['end'],
            "train_size": self.train_x.shape[0],
            "test_size": self.test_x.shape[0],
            "up_percent_change": file['up_percent_change'],
            "down_percent_change": file['down_percent_change']
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
            monitor='val_categorical_accuracy', 
            mode='max', 
            min_delta=0.001, 
            patience=ClassificationTraining.EARLY_STOPPING_PATIENCE
        )

        # Retrieve the Keras Model
        print("    1/6) Initializing Model...")
        model: Sequential = KerasModel(config=self.keras_model)

        # Compile the model
        print("    2/6) Compiling Model...")
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])
  
        # Train the model
        print("    3/6) Training Model...")
        history_object: History = model.fit(
            self.train_x,
            self.train_y,
            validation_split=0.2,
            batch_size=self.batch_size,
            epochs=ClassificationTraining.MAX_EPOCHS,
            callbacks=[ early_stopping ],
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        print("    4/6) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.test_x, self.test_y, verbose=0) # [loss, metric]

        # Save the model
        print("    5/6) Saving Model...")
        self._save_model(model)

        # Save the certificate
        print("    6/6) Saving Certificate...")
        certificate: IClassificationTrainingCertificate = self._save_certificate(
            start_time, 
            model, 
            history, 
            test_evaluation
        )

        # Return it so it can be added to the batch
        return certificate














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
            f.attrs['training_data_id'] = self.training_data_summary["id"]
            f.attrs['models'] = dumps(self.models)







    def _save_certificate(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float]
    ) -> IClassificationTrainingCertificate:
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

        Returns:
            IClassificationTrainingCertificate
        """
        # Build the certificate
        certificate: IClassificationTrainingCertificate = self._get_certificate(
            model, 
            start_time, 
            training_history, 
            test_evaluation
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
        test_evaluation: List[float]
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
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer._name,
            "loss": self.loss.name,
            "metric": self.metric.name,
            "batch_size": self.batch_size,
            "shuffle_data": self.shuffle_data,
            "keras_model_config": self.keras_model,

            # Training
            "training_start": start_time,
            "training_end": Utils.get_time(),
            "training_history": training_history,
            "test_evaluation": test_evaluation,

            # The configuration of the Classification
            "classification_config": {
                "id": self.id,
                "description": self.description,
                "training_data_id": self.training_data_summary["id"],
                "models": self.models,
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