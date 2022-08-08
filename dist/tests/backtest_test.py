import unittest
from modules._types import IBacktestConfig
from modules.database.Database import Database
from modules.model.ArimaModel import ArimaModel
from modules.model.RegressionModel import RegressionModel
from modules.model.ClassificationModel import ClassificationModel
from modules.backtest.Backtest import Backtest



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")



# Test Data
CONFIG: IBacktestConfig = {
    "id": "UNIT_TEST",
    "description": "Backtest Unit Test",
    "start": "1/10/2019",
    "end": "22/04/2022",
    "take_profit": 2.5,
    "stop_loss": 2.9,
    "idle_minutes_on_position_close": 120,
    "models": [
        {
            "id": "A212",
            "arima_models": [{
                "lookback": ArimaModel.DEFAULT_LOOKBACK,
                "predictions": ArimaModel.DEFAULT_PREDICTIONS,
                "arima": {"p": 2,"d": 1,"q": 2},
                "interpreter": ArimaModel.DEFAULT_INTERPRETER
            }]
        },
        {
            "id": "R_DNN_S3",
            "regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": RegressionModel.DEFAULT_INTERPRETER}]
        },
        {
            "id": "C_UNIT_TEST",
            "classification_models": [{"classification_id": "C_UNIT_TEST", "interpreter": ClassificationModel.DEFAULT_INTERPRETER}]
        },
        {
            "id": "CON_UNIT_TEST",    
            "arima_models": [{
                "lookback": ArimaModel.DEFAULT_LOOKBACK,
                "predictions": ArimaModel.DEFAULT_PREDICTIONS,
                "arima": {"p": 2,"d": 1,"q": 2},
                "interpreter": ArimaModel.DEFAULT_INTERPRETER
            }],
            "regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": RegressionModel.DEFAULT_INTERPRETER}],
            "classification_models": [{"classification_id": "C_UNIT_TEST", "interpreter": ClassificationModel.DEFAULT_INTERPRETER}],
            "consensus_model": { "interpreter": { "min_consensus": 2 } }
        }
    ]
}







## Test Class ##
class BacktestTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass




    # Can initialize a Backtest instance with any model and any settings
    def testInitialization(self):
        # Init a backtest instance
        backtest: Backtest = Backtest(CONFIG, test_mode=True)

        # Make sure the properties are equal
        self.assertEqual(backtest.id, CONFIG["id"])
        self.assertEqual(backtest.description, CONFIG["description"])
        self.assertEqual(len(backtest.models), len(CONFIG["models"]))
        self.assertEqual(backtest.take_profit, CONFIG["take_profit"])
        self.assertEqual(backtest.stop_loss, CONFIG["stop_loss"])
        self.assertEqual(backtest.idle_minutes_on_position_close, CONFIG["idle_minutes_on_position_close"])

        # Iterate over each model and make sure the IDs match
        for index, model in enumerate(backtest.models):
            if model.id != CONFIG["models"][index]["id"]:
                self.fail(f"Model ID Missmatch: {model.id}")




# Test Execution
if __name__ == '__main__':
    unittest.main()
