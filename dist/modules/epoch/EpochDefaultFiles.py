from modules.utils.Utils import Utils
from modules.epoch.EpochFile import EpochFile







# Default Files Creation
def create_default_files(epoch_id: str, regression_lookback: int, regression_predictions: int) -> None:
    """Creates all neccessary default files for the epoch to be able to
    get started.

    Args:
        epoch_id: str
            The name of the epoch.
    """
    # Create the backtest unit test file
    Utils.write(
        path=f"{epoch_id}/{EpochFile.EPOCH_PATH['backtest_configurations']}/UNIT_TEST.json", 
        indent=4,
        data={
            "id": "UNIT_TEST",
            "description": "The purpose of this test is to make sure the Backtest Module can run any Model.",
            "take_profit": 2.5,
            "stop_loss": 2.5,
            "idle_minutes_on_position_close": 30,
            "models": [
                { "id": "KR_UNIT_TEST", "keras_regressions": [{ "regression_id": "KR_UNIT_TEST" }] },
                #{ "id": "XGBR_UNIT_TEST", "xgb_regressions": [{ "regression_id": "XGBR_UNIT_TEST" }] },
                { "id": "KC_UNIT_TEST", "keras_classifications": [{ "classification_id": "KC_UNIT_TEST" }] },
                #{ "id": "XGBC_UNIT_TEST", "xgb_classifications": [{ "classification_id": "XGBC_UNIT_TEST" }] },
                {
                    "id": "CON_UNIT_TEST",
                    "keras_regressions": [{ "regression_id": "KR_UNIT_TEST" }],
                    #"xgb_regressions": [{ "regression_id": "XGBR_UNIT_TEST" }],
                    "keras_classifications": [{ "classification_id": "KC_UNIT_TEST" }],
                    #"xgb_classifications": [{ "classification_id": "XGBC_UNIT_TEST" }],
                    "consensus": { "interpreter": { "min_consensus": 2 } }
                }
            ]
        }
    )

    # Create the keras regression config unit test file
    Utils.write(
        path=f"{epoch_id}/{EpochFile.EPOCH_PATH['training_configs']}/keras_regression/UNIT_TEST/UNIT_TEST.json", 
        indent=4,
        data={
            "name": "KR_UNIT_TEST",
            "models": [
                {
                    "id": "KR_UNIT_TEST",
                    "description": "This is the official KerasRegressionModel for Unit Tests.",
                    "lookback": regression_lookback,
                    "predictions": regression_predictions,
                    "learning_rate": -1,
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
    )

    # Create the keras classification config unit test file
    Utils.write(
        path=f"{epoch_id}/{EpochFile.EPOCH_PATH['training_configs']}/keras_classification/UNIT_TEST/UNIT_TEST.json", 
        indent=4,
        data={
            "name": "KC_UNIT_TEST",
            "training_data_id": "FILL_THIS_VALUE", # Must fill once the training data has been generated
            "models": [
                {
                    "id": "KC_UNIT_TEST",
                    "description": "This is the official KerasClassificationModel for Unit Tests.",
                    "learning_rate": 0.001,
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
    )