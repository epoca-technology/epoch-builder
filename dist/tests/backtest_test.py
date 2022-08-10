import unittest
from modules._types import IBacktestConfig
from modules.database.Database import Database
from modules.backtest.Backtest import Backtest



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")



# Test Data
CONFIG: IBacktestConfig = {
    "id": "unit_test",
    "description": "The purpose of this test is to make sure the Backtest Module can run any Model.",
    "take_profit": 3,
    "stop_loss": 3,
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
