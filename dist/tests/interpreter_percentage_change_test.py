from typing import List
import unittest
from modules.utils import Utils
from modules.interpreter import PercentageChangeInterpreter



## Helpers ##


BASE: float = 100
def _get_preds(change: float, base: float = BASE) -> List[float]:
    """Retrieves a float list that includes the base as the first element 
    and then the changed version at the end.

    Args:
        change: float
            The percentage change that must be applied to the base
        base: float
            The base number that simulates the price

    Returns: 
        List[float]
    """
    return [base, base+1, base+2, base+1, Utils.alter_number_by_percentage(base, change)]






# Test Class
class PercentageChangeInterpreterTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    ## Interpreter Initialization ##



    # Can initialize an interpreter
    def testInitializeInterpreter(self):
        i = PercentageChangeInterpreter({"long": 1.5, "short": 1.5})
        self.assertEqual(i.long, 1.5)
        self.assertEqual(i.short, 1.5)



    # Cannot initialize an interpreter with invalid configuration
    def testInitializeInterpreterWithInvalidConfig(self):
        with self.assertRaises(ValueError):
            PercentageChangeInterpreter({"long": 0.001, "short": 1.5})
        with self.assertRaises(ValueError):
            PercentageChangeInterpreter({"long": 1.5, "short": 0.001})





    ## Predictions Interpretation ##



    # Can interpret a basic long
    def testBasicLongInterpretation(self):
        # Init the interpreter
        i = PercentageChangeInterpreter({"long": 0.5, "short": 0.5})

        # Can interpret a long position if the prediction change is equals
        result, description = i.interpret(_get_preds(0.5))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # Can interpret a long position if the prediction change is greater
        result, description = i.interpret(_get_preds(2.5))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.interpret(_get_preds(0.4))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')




    # Can interpret a basic short
    def testBasicShortInterpretation(self):
        # Init the interpreter
        i = PercentageChangeInterpreter({"long": 0.5, "short": 0.5})

        # Can interpret a short position if the prediction change is equals
        result, description = i.interpret(_get_preds(-0.5))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # Can interpret a short position if the prediction change is greater
        result, description = i.interpret(_get_preds(-2.5))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.interpret(_get_preds(-0.4))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')






    # Cannot interpret if invalid data is provided
    def testInterpretWithInvalidData(self):
        # Init the interpreter
        i = PercentageChangeInterpreter({"long": 0.5, "short": 0.5})
        with self.assertRaises(ValueError):
            i.interpret([1, 2, 3, 4])






# Test Execution
if __name__ == '__main__':
    unittest.main()