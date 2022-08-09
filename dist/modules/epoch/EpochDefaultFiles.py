from modules._types import IBacktestConfig, IKerasRegressionTrainingBatch, IKerasClassificationTrainingBatch
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
            "keras_regressions": [{ "regression_id": "KR_UNIT_TEST" }]
        },
        {
            "id": "KC_UNIT_TEST",
            "keras_classifications": [{ "classification_id": "KC_UNIT_TEST" }]
        },
        {
            "id": "CON_UNIT_TEST",
            "keras_regressions": [{ "regression_id": "KR_UNIT_TEST" }],
            "keras_classifications": [{ "classification_id": "KC_UNIT_TEST" }],
            "consensus": { "interpreter": { "min_consensus": 2 } }
        }
    ]
}




# Keras Regression Training Configs Unit Test
# This configuration trains the official unit test regression model for the current epoch.
# _EPOCH/model_assets/keras_regression_training_configs/UNIT_TEST.json
KERAS_REGRESSION_TRAINING_CONFIG_UT: IKerasRegressionTrainingBatch = {
    "name": "KR_UNIT_TEST",
    "models": [
        {
            "id": "KR_UNIT_TEST",
            "description": "This is the official KerasRegressionModel for Unit Tests.",
            "autoregressive": True,
            "lookback": 100,
            "predictions": 30,
            "optimizer": "adam",
            "loss": "mean_absolute_error",
            "metric": "mean_squared_error",
            "keras_model": {
                "name": "KR_DNN_S3",
                "units": [256, 128, 64],
                "activations": ["relu", "relu", "relu"]
            }
        }
    ]
}







# Keras Classification Training Config Unit Test
# This configuration trains the official unit test keras classification model for the current epoch.
# _EPOCH/model_assets/keras_classification_training_configs/UNIT_TEST.json
KERAS_CLASSIFICATION_TRAINING_CONFIG_UT: IKerasClassificationTrainingBatch = {
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
                "name": "KC_DNN_S3",
                "units": [256, 128, 64],
                "activations": ["relu", "relu", "relu"]
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

    # Create the keras classification config unit test file
    EpochFile.write(
        path=f"{epoch_id}/{EpochFile.MODEL_PATH['keras_classification_training_configs']}/UNIT_TEST.json", 
        data=KERAS_CLASSIFICATION_TRAINING_CONFIG_UT, 
        indent=4
    )