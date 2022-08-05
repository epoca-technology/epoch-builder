from modules.types import IBacktestConfig, ITrainingDataConfig
from modules.epoch.EpochFile import EpochFile




# Backtest Configuration Unit Test
# This configuration will run the Backtest process on each model type and output the results.
# _EPOCH/backtest_assets/configurations/unit_test.json
BACKTEST_CONFIG_UT: IBacktestConfig = {
    "id": "unit_test",
    "description": "The purpose of this test is to make sure the Backtest Module can run any Model.",
    "take_profit": 3,
    "stop_loss": 3,
    "idle_minutes_on_position_close": 30,
    "models": [
        {
            "id": "KR_UNIT_TEST",
            "regression_models": [{ "regression_id": "KR_UNIT_TEST" }]
        },
        {
            "id": "KC_UNIT_TEST",
            "classification_models": [{ "classification_id": "KC_UNIT_TEST" }]
        },
        {
            "id": "CON_UNIT_TEST",
            "regression_models": [{ "regression_id": "KR_UNIT_TEST" }],
            "classification_models": [{ "classification_id": "KC_UNIT_TEST" }],
            "consensus_model": { "interpreter": { "min_consensus": 2 } }
        }
    ]
}




# Keras Regression Training Configs Unit Test
# This configuration trains the official unit test regression model for the current epoch.
# _EPOCH/model_assets/keras_regression_training_configs/UNIT_TEST.json
KERAS_REGRESSION_TRAINING_CONFIG_UT = {
    "name": "KR_UNIT_TEST",
    "models": [
        {
            "id": "KR_UNIT_TEST",
            "description": "Official Unit Test Regression Model.",
            "autoregressive": True,
            "lookback": 100,
            "predictions": 30,
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metric": "mean_absolute_error",
            "keras_model": {
                "name": "R_UNIT_TEST",
                "units": [64],
                "activations": ["relu"]
            }
        }
    ]
}




# Classification Training Data Config Unit Test
# This configuration generates the training data that will be used to train test classification
# models.
# _EPOCH/model_assets/classification_training_data_configs/UNIT_TEST.json
CLASSIFICATION_TRAINING_DATA_CONFIG_UT: ITrainingDataConfig = {
    "regression_selection_id": "e5a03686-7bb9-4e2f-ab2f-3058281f589f", # Place Holder
    "description": "UNIT_TEST: DO NOT DELETE.",
    "steps": 5,
    "up_percent_change": 3,
    "down_percent_change": 3,
    "models": [
        { "id": "KR_UNIT_TEST","regression_models": [{ "regression_id": "KR_UNIT_TEST" }] }
    ],
    "include_rsi": True,
    "include_aroon": True
}







# Keras Classification Training Config Unit Test
# This configuration trains the official unit test keras classification model for the current epoch.
# _EPOCH/model_assets/keras_classification_training_configs/UNIT_TEST.json
KERAS_CLASSIFICATION_TRAINING_CONFIG_UT = {
    "name": "KC_UNIT_TEST",
    "training_data_id": "", # Must fill once the training data has been generated
    "models": [
        {
            "id": "KC_UNIT_TEST",
            "description": "This is the official KerasClassificationModel for Unit Tests.",
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metric": "binary_accuracy",
            "keras_model": {
                "name": "KC_UNIT_TEST",
                "units": [64],
                "activations": ["relu"]
            }
        }
    ]
}





# Default Files Creation
def create_default_files(epoch_id: str) -> None:
    """Creates all neccessary default files for the epoch to be able to
    get started.

    Args:
        epoch_id: str
            The name of the epoch.
    """
    # Create the backtest unit test file
    EpochFile.write(
        path=f"{epoch_id}/{EpochFile.BACKTEST_PATH['configurations']}/unit_test.json", 
        data=BACKTEST_CONFIG_UT, 
        indent=4
    )

    # Create the keras regression config unit test file
    EpochFile.write(
        path=f"{epoch_id}/{EpochFile.MODEL_PATH['keras_regression_training_configs']}/UNIT_TEST.json", 
        data=KERAS_REGRESSION_TRAINING_CONFIG_UT, 
        indent=4
    )

    # Create the classification training data unit test file
    EpochFile.write(
        path=f"{epoch_id}/{EpochFile.MODEL_PATH['classification_training_data_configs']}/UNIT_TEST.json", 
        data=CLASSIFICATION_TRAINING_DATA_CONFIG_UT, 
        indent=4
    )

    # Create the keras classification config unit test file
    EpochFile.write(
        path=f"{epoch_id}/{EpochFile.MODEL_PATH['keras_classification_training_configs']}/UNIT_TEST.json", 
        data=KERAS_CLASSIFICATION_TRAINING_CONFIG_UT, 
        indent=4
    )