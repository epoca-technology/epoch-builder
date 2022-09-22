from typing import List
from unittest import TestCase, main
from numpy import ndarray, array
from modules._types import IRegressionConfig, IRegressionTrainingCertificate
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.regression.Regression import Regression
from modules.regression.RegressionTraining import RegressionTraining





## Helpers ##



def _make_input_ds(indexes: List[int]) -> ndarray:
    """Builds a numpy array that can be used as input in order to generate
    predictions.

    Args:
        indexes: List[int]
            The list of random indexes to be used in order to generate the input ds.

    Returns:
        ndarray
    """
    features: List[List[float]] = []
    for i in indexes:
        features.append(Candlestick.NORMALIZED_PREDICTION_DF[["c"]].iloc[i:i+Epoch.REGRESSION_LOOKBACK, 0])
    return array(features)




def _validate_predictions(regression: Regression, preds: List[List[float]]) -> None:
    """Validates the generated predictions integrity.

    Args:
        regression: Regression
            The instance of the regression.
        preds: List[List[float]]
            The list of generated predictions.

    Raises:
        ValueError: 
            If any of the generated predictions is invalid in any way.
    """
    # Iterate over each prediction
    for pred in preds:
        # Validate the number of predictions
        if len(pred) != regression.predictions:
            print(pred)
            raise ValueError(f"The number of generated predictions is invalid: {len(pred)} != {regression.predictions}")
        
        # Validate the type of the predictions
        if not all(isinstance(x, float) for x in pred):
            print(pred)
            raise ValueError(f"Not all predicted values are valid floats.")

        # Make sure the predicted values are within acceptable ranges
        if not all(list(map(lambda x: x > 0 and x <= 1, preds[0]))):
            print(pred)
            raise ValueError("Some of the predictions are not within acceptable ranges (<= 0 or > 1)")






# TRAINING CERTIFICATE
MODEL_ID: str = "KR_UNIT_TEST"
CERT: IRegressionTrainingCertificate = RegressionTraining.get_certificate(MODEL_ID)
if CERT is None:
    raise RuntimeError("The unit test regression certificate could not be extracted. Please train the unit test regression prior to running the unit tests.")








## Test Class ##
class RegressionTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass





    # Initialize an instance with valid data and validate the integrity
    def testInitialize(self):
        # Initialize the instance
        r: Regression = Regression(MODEL_ID)

        # Output the config and make sure everything matches
        config: IRegressionConfig = r.get_config()
        self.assertEqual(config["id"], MODEL_ID)
        self.assertEqual(config["description"], CERT["description"])
        self.assertEqual(config["lookback"], CERT["regression_config"]["lookback"])
        self.assertEqual(config["predictions"], CERT["regression_config"]["predictions"])
        self.assertIsInstance(config["summary"], dict)







    # Can generate an individual prediction
    def testSinglePrediction(self):
        # Initialize the instance
        r: Regression = Regression(MODEL_ID)

        # Prepare the input dataset
        input_ds: ndarray = _make_input_ds([Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 1000])

        # Generate the individual prediction
        preds: List[float] = r.predict(input_ds)

        # The predictions should match the model's properties
        self.assertEqual(len(preds), 1)
        _validate_predictions(r, preds)





    # Can generate any number of predictions in one go
    def testMultiPrediction(self):
        # Initialize the instance
        r: Regression = Regression(MODEL_ID)

        # Prepare the input dataset
        input_ds: ndarray = _make_input_ds([
            Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 5000,
            Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 4000,
            Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 3000,
        ])

        # Generate the individual prediction
        preds: List[float] = r.predict(input_ds)

        # The predictions should match the model's properties
        self.assertEqual(len(preds), 3)
        _validate_predictions(r, preds)








    # Can generate an individual feature
    def testSingleFeature(self):
        # Initialize the instance
        r: Regression = Regression(MODEL_ID)

        # Prepare the input dataset
        input_ds: ndarray = _make_input_ds([Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 1000])

        # Generate the individual prediction
        features: List[float] = r.predict_feature(input_ds)

        # The predictions should match the model's properties
        self.assertEqual(len(features), 1)
        self.assertTrue(features[0] >= -1 and features[0] <= 1)







    # Can generate any number of features in one go
    def testMultiFeature(self):
        # Initialize the instance
        r: Regression = Regression(MODEL_ID)

        # Prepare the input dataset
        input_ds: ndarray = _make_input_ds([
            Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 5000,
            Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 4000,
            Candlestick.NORMALIZED_PREDICTION_DF.shape[0] - 3000,
        ])

        # Generate the individual prediction
        features: List[float] = r.predict_feature(input_ds)

        # The predictions should match the model's properties
        self.assertEqual(len(features), 3)
        self.assertTrue(features[0] >= -1 and features[0] <= 1)
        self.assertTrue(features[1] >= -1 and features[1] <= 1)
        self.assertTrue(features[2] >= -1 and features[2] <= 1)





# Test Execution
if __name__ == '__main__':
    main()
