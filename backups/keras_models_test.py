import unittest
from typing import List, Any
from numpy import array
from keras import Sequential
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import MeanSquaredError as MeanSquaredErrorMetric, MeanAbsoluteError as MeanAbsoluteErrorMetric
from keras.optimizers import adam_v2, rmsprop_v2
from modules.keras_models import KerasModel




# Test Class
class KerasModelsTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass




    ## Regression Models ##

    # Dense


    # CNN


    # LSTM

    # LSTM_STACKLESS
    def test_LSTM_STACKLESS(self):
        model: Sequential = KerasModel('regression', {
            "name": "LSTM_MEDIUM_STACK_BALANCED_DROPOUT",
            "units": [80, 80, 80, 80],
            "dropout_rates": [0.2, 0.2, 0.2],
            "predictions": 5
        })

        # Compile the model
        model.compile(loss=MeanAbsoluteError(), optimizer=adam_v2.Adam(), metrics=[MeanSquaredErrorMetric()])
        model.build(input_shape=(1, 50, 5))


        print("\nModel:")
        print("Type:", model.__class__.__name__)
        print("Optimizer:", model.optimizer.get_config())
        print("Loss:", model.loss.get_config())
        print("Metric:", model._metrics)
        print("Input Shape:", model.input_shape)
        print("Output Shape:", model.output_shape)


        print("Weights Length:", len(model.get_weights()))
        """weights = model.get_weights()
        for w in weights:
            print(len(w))"""

        print("\n\nLayers:")
        trainable_params: int = 0
        non_trainable_params: int = 0
        for layer in model.layers:
            layer_params: int = layer.count_params()
            print("\n", layer.name)
            print("Params:", layer_params)
            print("Input Shape:", layer.input_shape)
            print("Output Shape:", layer.output_shape)
            print("Trainable:", layer.trainable)
            if layer.trainable:
                trainable_params = trainable_params + layer_params
            else:
                non_trainable_params = non_trainable_params + layer_params


        print("\n\nModel Params:")
        print("Total: ", model.count_params())
        print("Trainable: ", trainable_params)
        print("Non-Trainable: ", non_trainable_params)

        print("\n\nSummary:\n")
        print(model.summary())




    ## Autoregressive Regression Models ##






    ## Classification Models ##




# Test Execution
if __name__ == '__main__':
    unittest.main()