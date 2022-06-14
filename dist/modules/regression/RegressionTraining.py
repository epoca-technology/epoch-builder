from typing import Union, Tuple, List
from os import makedirs
from os.path import exists
from random import randint
from numpy import mean
from pandas import DataFrame
from json import dumps
from tqdm import tqdm
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5
from keras import Sequential
from keras.optimizers import adam_v2 as adam, rmsprop_v2 as rmsprop
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import MeanSquaredError as MeanSquaredErrorMetric, MeanAbsoluteError as MeanAbsoluteErrorMetric
from keras.callbacks import EarlyStopping, History
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.keras_models import KerasModel, IKerasModelConfig, IKerasModelTrainingHistory, get_summary, KERAS_PATH
from modules.regression import IRegressionTrainingConfig, TrainingWindowGenerator, IRegressionTrainingCertificate, \
    Regression, IRegressionEvaluation




class RegressionTraining:
    """RegressionTraining Class

    This class handles the training of a RegressionModel.

    Class Properties:
        EARLY_STOPPING_PATIENCE: int
            The number of epochs it will allow to be executed without showing an improvement.
        MAX_EPOCHS: int
            The maximum amount of epochs the training process can go through.
        DEFAULT_MAX_EVALUATIONS: int
            The default maximum number of evaluations that will be performed on the trained model.

    Instance Properties:
        test_mode: bool
             If running from unit tests, it won't check the model's directory.
        hyperparams_mode: bool
            If enabled, it means that the purpose of the training is to identify the best hyperparams
            and therefore, a large amount of models will be trained.
        id: str
            A descriptive identifier compatible with filesystems
        model_path: str
            The directory in which the model will be stored.
        description: str
            Important information regarding the model that will be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        learning_rate: float
            The learning rate to be used by the optimizer.
        optimizer: Union[adam.Adam, rmsprop.RMSProp]                    "adam"|"rmsprop"
            The optimizer that will be used to train the model.
        loss: Union[MeanSquaredError, MeanAbsoluteError]                "mean_squared_error"|"mean_absolute_error"
            The loss function that will be used for training.
        metric: Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric]  "mean_squared_error"|"mean_absolute_error"
            The metric function that will be used for training.
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
    # The max number of training epochs that can occur without showing improvements.
    EARLY_STOPPING_PATIENCE: int = 5

    # The maximum number of EPOCHs a model can go through during training
    MAX_EPOCHS: int = 10

    # The max number of evaluations that will be performed on the trained regression model.
    # Notice that if the number of evals is much smaller than the max it means there could be
    # an irregularity with the model as the predicted changes are under 0.05%
    DEFAULT_MAX_EVALUATIONS: int = 20



    ## Initialization ##




    def __init__(
        self, 
        config: IRegressionTrainingConfig, 
        max_evaluations: Union[int, None],
        hyperparams_mode: bool=False,
        test_mode: bool = False
    ):
        """Initializes the RegressionTraining Instance.

        Args:
            config: IRegressionTrainingConfig
                The configuration that will be used to train the model.
            max_evaluations: Union[int, None]
                The maximum number of evaluations that can be performed
            hyperparams_mode: bool
                If enabled, there will be no verbosity during training and eval.
            test_mode: bool
                If running from unit tests, it won't check the model's directory.

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
        self.id: str = config['id']
        if self.id[0:2] != 'R_':
            raise ValueError("The ID of the Regression Model must be preffixed with R_")

        # Initialize the Model's path
        self.model_path: str = f"{KERAS_PATH['models']}/{self.id}"

        # Initialize the description
        self.description: str = config['description']

        # Initialize the lookback
        self.lookback: int = config['lookback']

        # Initialize the predictions output
        self.predictions: int = config['predictions']

        # Initialize the Learning Rate
        self.learning_rate: float = config['learning_rate']

        # Initialize the optimizer function
        self.optimizer: Union[adam.Adam, rmsprop.RMSProp] = self._get_optimizer(config["optimizer"])

        # Initialize the loss function
        self.loss: Union[MeanSquaredError, MeanAbsoluteError] = self._get_loss(config['loss'])

        # Initialize the metric function
        self.metric: Union[MeanSquaredErrorMetric, MeanAbsoluteErrorMetric] = self._get_metric(config['metric'])

        # Initialize the Keras Model's Configuration and populate the lookback
        self.keras_model: IKerasModelConfig = config["keras_model"]
        self.keras_model["lookback"] = self.lookback

        # Split the candlesticks into train, val and test
        train_df, val_df, test_df = self._get_data()

        # Initialize the Window Instance
        self.window: TrainingWindowGenerator = TrainingWindowGenerator({
            "input_width": self.lookback,
            "label_width": 1,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "label_columns": ["c"]
        })

        # Initialize the Dataset Sizes
        self.train_size: int = train_df.shape[0]
        self.val_size: int = val_df.shape[0]
        self.test_size: int = test_df.shape[0]

        # Initialize the max evaluations
        self.max_evaluations: int = max_evaluations if isinstance(max_evaluations, int) else RegressionTraining.DEFAULT_MAX_EVALUATIONS

        # Initialize the model's directory if not unit testing
        if not self.test_mode:
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
        if func_name == 'adam':
            return adam.Adam(learning_rate=self.learning_rate)
        elif func_name == 'rmsprop':
            return rmsprop.RMSProp(learning_rate=self.learning_rate)
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
        if func_name == 'mean_squared_error':
            return MeanSquaredErrorMetric()
        elif func_name == 'mean_absolute_error':
            return MeanAbsoluteErrorMetric()
        else:
            raise ValueError(f"The metric function for {func_name} was not found.")






    



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
            patience=RegressionTraining.EARLY_STOPPING_PATIENCE,
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
            self.window.train, 
            validation_data=self.window.val,
            epochs=RegressionTraining.MAX_EPOCHS,
            callbacks=[ early_stopping ],
            shuffle=False,
            verbose=0
        )

        # Initialize the Training History
        history: IKerasModelTrainingHistory = history_object.history

        # Evaluate the test dataset
        if not self.hyperparams_mode:
            print("    4/7) Evaluating Test Dataset...")
        test_evaluation: List[float] = model.evaluate(self.window.test, verbose=0) # [loss, metric]

        # Save the model
        if not self.hyperparams_mode:
            print("    5/7) Saving Model...")
        self._save_model(model)

        # Perform the regression evaluation
        regression_evaluation: IRegressionEvaluation = self._evaluate_regression()

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













    ## Regression Evaluation ##




    def _evaluate_regression(self) -> IRegressionEvaluation:
        """Loads the trained model that has just been saved and performs a series
        of evaluations on random sequences based on the model's config.

        Returns:
            IRegressionEvaluation
        """
        # Initialize the Regression Instance
        regression: Regression = Regression(self.id)

        # Init the min and max values for the random candlestick index
        min_i: int = 0
        max_i: int = int(Candlestick.DF.shape[0] * 0.95) # Omit the tail to prevent index errors

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
            progress_bar.set_description("    6/7) Evaluating Regression")

        # Perform the evaluation
        for i in range(self.max_evaluations):
            # Generate a random candlestick ot
            random_ot: float = Candlestick.DF.iloc[randint(min_i, max_i)]['ot']

            # Retrieve the normalized lookback df
            lookback_df: DataFrame = Candlestick.get_lookback_df(regression.lookback, random_ot, normalized=True)

            # Generate the predictions and calculate the % change in price
            preds: List[float] = regression.predict(lookback_df)
            preds_change: float = Utils.get_percentage_change(preds[0], preds[-1])

            # Check if there was a non-neutral change
            if preds_change >= 0.05 or preds_change <= -0.05:
                # Retrieve the close price of the last candlestick in the window. This is the last candlestick
                # in the lookback_df[-1] + self.predictions subset.
                window_cp: float = \
                    Candlestick.NORMALIZED_PREDICTION_DF[
                        Candlestick.PREDICTION_DF['ot'] > random_ot
                    ].iloc[regression.predictions - 1:regression.predictions].iloc[0]['c']

                # Calculate the actual change in price from the last lookback candlestick and the 
                # actual window close price
                real_change: float = Utils.get_percentage_change(lookback_df.iloc[-1]['c'], window_cp)

                # Check if an increase was predicted and evaluate the outcome
                if preds_change > 0:
                    increase.append(preds_change)
                    if real_change > 0:
                        increase_successful.append(preds_change)
                        increase_outcomes += 1
                    else:
                        decrease_outcomes += 1

                # Check if a decrease was predicted and evaluate the outcome
                elif preds_change < 0:
                    decrease.append(preds_change)
                    if real_change < 0:
                        decrease_successful.append(preds_change)
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
            f.attrs['lookback'] = self.lookback
            f.attrs['predictions'] = self.predictions







    def _save_certificate(
        self, 
        start_time: int, 
        model: Sequential, 
        training_history: IKerasModelTrainingHistory, 
        test_evaluation: List[float],
        regression_evaluation: IRegressionEvaluation
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
            regression_evaluation: IRegressionEvaluation
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
        regression_evaluation: IRegressionEvaluation
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
            regression_evaluation: IRegressionEvaluation
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
            "val_size": self.val_size,
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

            # Post Training Evaluation
            "regression_evaluation": regression_evaluation,

            # The configuration of the Regression
            "regression_config": {
                "id": self.id,
                "description": self.description,
                "lookback": self.lookback,
                "predictions": self.predictions,
                "summary": get_summary(model)
            },
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