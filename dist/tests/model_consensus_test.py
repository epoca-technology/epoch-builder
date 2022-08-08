import unittest
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
    "arima_models": [{"lookback": 150,"predictions": 10,"arima": { "p": 2, "d": 1, "q": 2 },"interpreter": { "long": 0.5, "short": 0.5 }}],
    "regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": { "long": 1, "short": 1 }}],
    "classification_models": [{"classification_id": "C_UNIT_TEST", "interpreter": { "min_probability": 0.51 }}],
    "consensus_model": { "interpreter": { "min_consensus": 2 } }
}







## Test Class ##
class ConsensusModelTestCase(unittest.TestCase):
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
        time: int = Candlestick.DF.iloc[655858]["ot"]

        # Perform a random prediction
        pred: IPrediction = model.predict(time)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 3)

        # Retrieve the summary and make sure it is valid
        summary: IModel = model.get_model()
        self.assertIsInstance(summary, dict)
        self.assertEqual(len(summary["arima_models"]), 1)
        self.assertEqual(len(summary["regression_models"]), 1)
        self.assertEqual(len(summary["classification_models"]), 1)
        self.assertEqual(len(summary["consensus_model"]["sub_models"]), 3)
        self.assertDictEqual(summary["consensus_model"]["interpreter"], CONFIG["consensus_model"]["interpreter"])








# Test Execution
if __name__ == '__main__':
    unittest.main()
