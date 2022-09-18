from typing import Tuple
from unittest import TestCase, main
from copy import deepcopy
from numpy import ndarray, concatenate
from modules._types import IKerasModelConfig, IRegressionTrainingConfig, IRegressionTrainAndTestDatasets
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.regression.RegressionTraining import RegressionTraining



# Regression Parameters
lookback: int = Epoch.REGRESSION_LOOKBACK
predictions: int = Epoch.REGRESSION_PREDICTIONS



# Dataset Validator
# Makes sure the built dataset matches the original dataframe.
def validate_dataset(features: ndarray, labels: ndarray) -> None:
    # If they have different lens, something is wrong
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Features and labels do not have the same number of rows. {features.shape[0]} != {labels.shape[0]}")

    # Iterate over each index and make sure that features and labels are correct
    for i in range(features.shape[0]):
        # Compare the features
        expected_features: ndarray = Candlestick.NORMALIZED_PREDICTION_DF["c"].iloc[i:lookback+i].to_numpy()
        if not (expected_features == features[i]).all():
            print(f"Head: {expected_features[0]} == {features[i][0]}")
            print(f"Tail: {expected_features[-1]} == {features[i][-1]}")
            print(f"Len: {len(expected_features)} == {len(features[i])}")
            raise ValueError(f"Regression Dataset Feature Discrepancy on index: {i}")

        # Compare the labels
        expected_labels: ndarray = Candlestick.NORMALIZED_PREDICTION_DF["c"].iloc[lookback+i:lookback+i+predictions].to_numpy()
        if not (expected_labels == labels[i]).all():
            print(f"Head: {expected_labels[0]} == {labels[i][0]}")
            print(f"Tail: {expected_labels[-1]} == {labels[i][-1]}")
            print(f"Len: {len(expected_labels)} == {len(labels[i])}")
            raise ValueError(f"Regression Dataset Label Discrepancy on index: {i}")
    




# Dataset Size
# In order to train a model, the dataset must be split into train and test arrays.
# The train array is use to train the model and the test is used to evaluate it.
def calculate_dataset_estimated_sizes() -> Tuple[int, int]:
    """Calculates the estimate size of the train and test dataset.

    Returns:
        Tuple[int, int]
        (train_size, test_size)
    """
    return int(Candlestick.NORMALIZED_PREDICTION_DF.shape[0] * Epoch.TRAIN_SPLIT), \
        int(Candlestick.NORMALIZED_PREDICTION_DF.shape[0] * (1 - Epoch.TRAIN_SPLIT))





# Default Training Configuration
CONFIG: IRegressionTrainingConfig = {
    "id": "KR_UNIT_TEST",
    "description": "This is the official KerasRegressionModel for Unit Tests.",
    "lookback": lookback,
    "predictions": predictions,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "mean_absolute_error",
    "metric": "mean_squared_error",
    "keras_model": {
        "name": "KR_DNN_S3",
        "units": [256, 128, 64],
        "activations": ["relu", "relu", "relu"]
    }
}










## Test Class ##
class RegressionTrainingTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    # Can build the train and test dataset and then initialize a valid training instance
    def testInitializeFlow(self):
        # Make the datasets and unpack them
        datasets: IRegressionTrainAndTestDatasets = RegressionTraining.make_train_and_test_datasets()
        train_x, train_y, test_x, test_y = datasets

        # Make sure the features and the labels have the same number of items
        self.assertEqual(train_x.shape[0], train_y.shape[0])
        self.assertEqual(test_x.shape[0], test_y.shape[0])

        # Calculate the dataset sizes
        train_size, test_size = calculate_dataset_estimated_sizes()

        # Make sure the dataset split was applied correctly
        self.assertAlmostEqual(train_x.shape[0], train_size, delta=200)
        self.assertAlmostEqual(test_x.shape[0], test_size, delta=200)

        # Finally, validate the entire dataset
        validate_dataset(features=concatenate((train_x, test_x)), labels=concatenate((train_y, test_y)))

        # Init the config
        config: IRegressionTrainingConfig = deepcopy(CONFIG)

        # Initialize the instance
        training: RegressionTraining = RegressionTraining(config, datasets=datasets)

        # Validate properties
        self.assertEqual(training.id, config["id"])
        self.assertEqual(training.description, config["description"])
        self.assertEqual(training.lookback, config["lookback"])
        self.assertEqual(training.predictions, config["predictions"])
        self.assertEqual(training.learning_rate, config["learning_rate"])
        self.assertEqual(training.optimizer._name.lower(), config["optimizer"])
        self.assertEqual(training.loss.name, config["loss"])
        expected_keras_model: IKerasModelConfig = deepcopy(config["keras_model"])
        expected_keras_model["lookback"] = config["lookback"]
        expected_keras_model["predictions"] = config["predictions"]
        self.assertDictEqual(training.keras_model, expected_keras_model)
        self.assertIsInstance(training.train_x, ndarray)
        self.assertIsInstance(training.train_y, ndarray)
        self.assertIsInstance(training.test_x, ndarray)
        self.assertIsInstance(training.test_y, ndarray)
        self.assertTrue(len(training.train_x) > 0)
        self.assertTrue(len(training.train_y) > 0)
        self.assertTrue(len(training.test_x) > 0)
        self.assertTrue(len(training.test_y) > 0)






# Test Execution
if __name__ == '__main__':
    main()
