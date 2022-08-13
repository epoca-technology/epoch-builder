from unittest import TestCase, main
from modules._types import IModel, IPrediction
from modules.database.Database import Database
from modules.candlestick.Candlestick import Candlestick
from modules.model.ConsensusModel import ConsensusModel





## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TRAINING DATA FILE
CONFIG: IModel = {
    "id": "CON_UNIT_TEST",
    "keras_regressions": [{ "regression_id": "KR_UNIT_TEST" }],
    #"xgb_regressions": [{ "regression_id": "XGBR_UNIT_TEST" }],
    "keras_classifications": [{ "classification_id": "KC_UNIT_TEST" }],
    #"xgb_classifications": [{ "classification_id": "XGBC_UNIT_TEST" }],
    "consensus": { "interpreter": { "min_consensus": 2 } }
}







## Test Class ##
class ConsensusModelTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    ## Initialization ##


    # Initialize an instance with valid data and validate the integrity
    def testInitialize(self):
        # Initialize the instance
        model: ConsensusModel = ConsensusModel(CONFIG)

        # Make sure the instance is recognized
        self.assertIsInstance(model, ConsensusModel)
        self.assertEqual(type(model).__name__, "ConsensusModel")

        # Init the test candlestick time
        time: int = Candlestick.DF.iloc[55858]["ot"]

        # Perform a random prediction
        pred: IPrediction = model.predict(time)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 2)

        # Retrieve the summary and make sure it is valid
        summary: IModel = model.get_model()
        self.assertIsInstance(summary, dict)
        self.assertEqual(len(summary["keras_regressions"]), 1)
        self.assertEqual(len(summary["keras_classifications"]), 1)
        #self.assertEqual(len(summary["xgb_regressions"]), 1)
        #self.assertEqual(len(summary["xgb_classifications"]), 1)
        self.assertEqual(len(summary["consensus"]["sub_models"]), 2)
        self.assertDictEqual(summary["consensus"]["interpreter"], CONFIG["consensus"]["interpreter"])








# Test Execution
if __name__ == '__main__':
    main()
