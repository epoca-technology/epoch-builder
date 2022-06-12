from typing import Union, List, Tuple
from os import makedirs
from os.path import exists
from random import randint
from numpy import mean
from pandas import DataFrame, Series, concat
from json import dumps
from tqdm import tqdm
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, History
from modules.utils import Utils
from modules.candlestick import Candlestick
from modules.model import IModel, RegressionModelFactory, ArimaModel, RegressionModel, ClassificationModel, IPrediction
from modules.keras_models import KerasModel, IKerasModelConfig, IKerasModelTrainingHistory, get_summary, KERAS_PATH
from modules.classification import ITrainingDataFile, ICompressedTrainingData, decompress_training_data, \
    IClassificationTrainingConfig, ITrainingDataSummary, IClassificationEvaluation, IClassificationTrainingCertificate
        




class ClassificationTraining:
    """ClassificationTraining Class

    This class handles the training of a Classification Model.

    Class Properties:
        TRAIN_SPLIT: float
            The split that will be used on the complete df in order to generate train and test 
                features & labels dfs
        EARLY_STOPPING_PATIENCE: int
            The number of epochs it will allow to be executed without showing a performance.
        MAX_EPOCHS: int
            The maximum amount of epochs the training process can go through.
        DEFAULT_MAX_EVALUATIONS: int
            The default maximum number of evaluations that will be performed on the trained model.

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks.
        hyperparams_mode: bool
            If enabled, it means that the purpose of the training is to identify the best hyperparams
            and therefore, a large amount of models will be trained.
        id: str
            A descriptive identifier compatible with filesystems.
        model_path: str
            The directory in which the model will be stored.
        description: str
            Important information regarding the model that will be trained.
        models: List[IModel]
            The list of ArimaModel|RegressionModel used to generate the training data.
        regressions: List[Union[ArimaModel, RegressionModel]]
            The list of regression model instances.
        learning_rate: float
            The learning rate to be used by the optimizer. If None is provided it uses the default
            one.   
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
        max_evaluations: int
            The Maximum number of evaluations that will be performed post-training. If none is provided, 
            the default value will be used.
    """
    # Train and Test DataFrame Split
    TRAIN_SPLIT: float = 0.8

    # The max number of training epochs that can occur without showing improvements.
    EARLY_STOPPING_PATIENCE: int = 50

    # The maximum number of EPOCHs a model can go through during training
    MAX_EPOCHS: int = 1000

    # The max number of evaluations that will be performed on the trained classification model.
    # Notice that if the number of evals is much smaller than the max it means there could be
    # an irregularity with the model as the probabilities are too close to the 50%.
    DEFAULT_MAX_EVALUATIONS: int = 350



    ## Initialization ##



    def __init__(
        self, 
        training_data_file: ITrainingDataFile, 
        config: IClassificationTrainingConfig, 
        max_evaluations: Union[int, None],
        hyperparams_mode: bool=False,
        test_mode: bool = False
    ):
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
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the mode
        self.hyperparams_mode: bool = hyperparams_mode
        
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
        self.regressions: List[Union[ArimaModel, RegressionModel]] = [ RegressionModelFactory(m) for m in self.models ]

        # Initialize the Learning Rate
        self.learning_rate: float = config["learning_rate"]

        # Initialize the optimizer function
        self.optimizer: Union[adam.Adam, rmsprop.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[CategoricalCrossentropy, BinaryCrossentropy] = self._get_loss(config["loss"])

        # Initialize the metric function
        self.metric: Union[CategoricalAccuracy, BinaryAccuracy] = self._get_metric(config["metric"])

        # Initialize the Training Data
        train_x, train_y, test_x, test_y = self._get_data(training_data_file["training_data"])
        self.train_x: DataFrame = train_x
        self.train_y: DataFrame = train_y
        self.test_x: DataFrame = test_x
        self.test_y: DataFrame = test_y

        # Initialize the Training Data Summary
        self.training_data_summary: ITrainingDataSummary = self._get_training_data_summary(training_data_file)

        # Initialize the Keras Model's Configuration
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["features_num"] = self.training_data_summary["features_num"]

        # Initialize the candlesticks if not unit testing
        if not self.test_mode:
            Candlestick.init(max([m.get_lookback() for m in self.regressions]))

        # Initialize the max evaluations
        self.max_evaluations: int = max_evaluations if isinstance(max_evaluations, int) else ClassificationTraining.DEFAULT_MAX_EVALUATIONS

        # Initialize the model's directory if not unit testing
        if not self.test_mode:
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
        train_y: DataFrame = concat([train_x.pop(x) for x in ["up", "down"]], axis=1)
        test_y: DataFrame = concat([test_x.pop(x) for x in ["up", "down"]], axis=1)

        # Return the packed dfs
        return train_x, train_y, test_x, test_y






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
        if func_name == "adam":
            return adam.Adam(learning_rate=self.learning_rate)
        elif func_name == "rmsprop":
            return rmsprop.RMSProp(learning_rate=self.learning_rate)
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
            patience=ClassificationTraining.EARLY_STOPPING_PATIENCE,
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
            epochs=ClassificationTraining.MAX_EPOCHS,
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
        classification_evaluation: IClassificationEvaluation = self._evaluate_classification()

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















    ## Classification Evaluation ##




    def _evaluate_classification(self) -> IClassificationEvaluation:
        """Loads the trained model that has just been saved and performs a series
        of evaluations on random sequences.

        Returns:
            IClassificationEvaluation
        """
        # Initialize the ClassificationModel Instance
        classification: ClassificationModel = ClassificationModel({
            "id": self.id,
            "classification_models": [{ "classification_id": self.id, "interpreter": { "min_probability": 0.51 }}]
        })

        # Init the min and max values for the random candlestick indexes
        min_i: int = 0
        max_i: int = int(Candlestick.DF.shape[0] * 0.99) # Omit the tail to prevent index errors

        # Init evaluation data
        evals: int = 0
        increase: List[float] = []
        increase_successful: List[float] = []
        decrease: List[float] = []
        decrease_successful: List[float] = []
        increase_outcomes: int = 0
        decrease_outcomes: int = 0

        # Init the progress bar
        if not self.hyperparams_mode:
            progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=self.max_evaluations)
            progress_bar.set_description("    6/7) Evaluating Classification")

        # Perform the evaluation
        for i in range(self.max_evaluations):
            # Generate a random index and initialize the random start candlestick
            random_index: int = randint(min_i, max_i)
            candlestick: Series = Candlestick.DF.iloc[random_index]

            # Generate a perdiction
            pred: IPrediction = classification.predict(candlestick["ot"], enable_cache=False)

            # Check if it is a non-neutral prediction
            if pred["r"] != 0:
                # Retrieve the outcome of the evaluation
                outcome: int = self._get_evaluation_outcome(random_index, candlestick)

                # Only process the evaluation if the outcome was determined
                if outcome != 0:
                    # Check if the Classification predicted an increase
                    if pred["r"] == 1:
                        # Append the increase prediction to the list
                        increase.append(float(pred["md"][0]["up"]))
                        
                        # Check if the prediction was correct
                        if outcome == 1:
                            increase_successful.append(float(pred["md"][0]["up"]))
                            increase_outcomes += 1
                        else:
                            decrease_outcomes += 1

                    # Otherwise, the Classification predicted a decrease
                    else:
                        # Append the decrease prediction to the list
                        decrease.append(float(pred["md"][0]["dp"]))
                        
                        # Check if the prediction was correct
                        if outcome == -1:
                            decrease_successful.append(float(pred["md"][0]["dp"]))
                            decrease_outcomes += 1
                        else:
                            increase_outcomes += 1

                    # Increment the evals performed
                    evals += 1

            # Update the progress bar
            if not self.hyperparams_mode:
                progress_bar.update()

        # Initialize the lens for performance
        increase_num: int = len(increase)
        increase_successful_num: int = len(increase_successful)
        decrease_num: int = len(decrease)
        decrease_successful_num: int = len(decrease_successful)

        # Finally, return the results
        return {
            # Evaluations
            "evaluations": evals,
            "max_evaluations": self.max_evaluations,

            # Prediction counts
            "increase_num": increase_num,
            "increase_successful_num": increase_successful_num,
            "decrease_num": decrease_num,
            "decrease_successful_num": decrease_successful_num,

            # Accuracy
            "increase_acc": Utils.get_percentage_out_of_total(increase_successful_num, increase_num if increase_num > 0 else 1),
            "decrease_acc": Utils.get_percentage_out_of_total(decrease_successful_num, decrease_num if decrease_num > 0 else 1),
            "acc": Utils.get_percentage_out_of_total(increase_successful_num+decrease_successful_num, evals if evals > 0 else 1),
            
            # Predictions Overview 
            "increase_list": increase,
            "increase_max": max(increase if increase_num > 0 else [0]),
            "increase_min": min(increase if increase_num > 0 else [0]),
            "increase_mean": mean(increase if increase_num > 0 else [0]),
            "increase_successful_list": increase_successful,
            "increase_successful_max": max(increase_successful if increase_successful_num > 0 else [0]),
            "increase_successful_min": min(increase_successful if increase_successful_num > 0 else [0]),
            "increase_successful_mean": mean(increase_successful if increase_successful_num > 0 else [0]),
            "decrease_list": decrease,
            "decrease_max": max(decrease if decrease_num > 0 else [0]),
            "decrease_min": min(decrease if decrease_num > 0 else [0]),
            "decrease_mean": mean(decrease if decrease_num > 0 else [0]),
            "decrease_successful_list": decrease_successful,
            "decrease_successful_max": max(decrease_successful if decrease_successful_num > 0 else [0]),
            "decrease_successful_min": min(decrease_successful if decrease_successful_num > 0 else [0]),
            "decrease_successful_mean": mean(decrease_successful if decrease_successful_num > 0 else [0]),

            # Outcomes
            "increase_outcomes": increase_outcomes,
            "decrease_outcomes": decrease_outcomes,
        }












    def _get_evaluation_outcome(self, random_index: int, start_candlestick: Series) -> int:
        """Simulates a trading position that starts at a random candlestick and iterates 
        over the next records until an outcome is found.

        Args:
            random_index: int
                The random index generated for the evaluation.
            start_candlestick: Series
                The candlestick located at the random_index.

        Returns:
            int
            1 (Increase) | -1 (Decrease)
        """
        # Initialize the outcome
        outcome: int = 0

        # Initialize the price range
        increase_price: float = Utils.alter_number_by_percentage(start_candlestick["o"], self.training_data_summary["up_percent_change"])
        decrease_price: float = Utils.alter_number_by_percentage(start_candlestick["o"], -self.training_data_summary["down_percent_change"])

        # Iterate over the next candlesticks until the outcome is discovered
        candlestick_index: int = random_index + 1
        while outcome == 0 and candlestick_index < int(Candlestick.DF.shape[0]*0.99):
            # Initialize the candlestick
            candlestick: Series = Candlestick.DF.iloc[candlestick_index]

            # Check if it is an increase
            if candlestick["h"] >= increase_price:
                outcome = 1

            # Check if it is a decrease
            elif candlestick["l"] <= decrease_price:
                outcome = -1

            # Increment the index and iterate again
            candlestick_index += 1

        # Finally, return the outcome
        return outcome















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
        classification_evaluation: IClassificationEvaluation
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
            classification_evaluation: IClassificationEvaluation
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
        classification_evaluation: IClassificationEvaluation
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
            classification_evaluation: IClassificationEvaluation
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