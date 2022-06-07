import unittest
from modules.backtest import Backtest, IBacktestConfig


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
            "arima_models": [{"arima": {"p": 2,"d": 1,"q": 2}}]
        },
        {
            "id": "R_DNN_S3",
            "regression_models": [{"regression_id": "R_UNIT_TEST","interpreter": {"long": 0.05,"short": 0.05}}]
        },
        {
            "id": "C_UNIT_TEST",
            "classification_models": [{"classification_id": "C_UNIT_TEST","interpreter": { "min_probability": 0.51 }}]
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
