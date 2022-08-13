from typing import List
from unittest import TestCase, main
from modules.database.Database import Database
from modules.utils.Utils import Utils
from modules.interpreter.PercentageChangeInterpreter import PercentageChangeInterpreter




## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





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
class PercentageChangeInterpreterTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    ## Interpreter Initialization ##



    # Can initialize an interpreter
    def testInitializeInterpreter(self):
        i = PercentageChangeInterpreter({"min_increase_change": 1.5, "min_decrease_change": -1.5})
        self.assertEqual(i.min_increase_change, 1.5)
        self.assertEqual(i.min_decrease_change, -1.5)



    # Cannot initialize an interpreter with invalid configuration
    def testInitializeInterpreterWithInvalidConfig(self):
        with self.assertRaises(ValueError):
            PercentageChangeInterpreter({"min_increase_change": 0.001, "min_decrease_change": 1.5})
        with self.assertRaises(ValueError):
            PercentageChangeInterpreter({"min_increase_change": 1.5, "min_decrease_change": -0.001})





    ## Predictions Interpretation ##



    # Can interpret a basic min_increase_change
    def testBasicLongInterpretation(self):
        # Init the interpreter
        i = PercentageChangeInterpreter({"min_increase_change": 1, "min_decrease_change": -1})

        # Can interpret a min_increase_change position if the prediction change is equals
        result, description = i.interpret(_get_preds(1))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # Can interpret a min_increase_change position if the prediction change is greater
        result, description = i.interpret(_get_preds(2.5))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.interpret(_get_preds(0.9))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')




    # Can interpret a basic min_decrease_change
    def testBasicShortInterpretation(self):
        # Init the interpreter
        i = PercentageChangeInterpreter({"min_increase_change": 1, "min_decrease_change": -1})

        # Can interpret a min_decrease_change position if the prediction change is equals
        result, description = i.interpret(_get_preds(-1))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # Can interpret a min_decrease_change position if the prediction change is greater
        result, description = i.interpret(_get_preds(-2.5))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.interpret(_get_preds(-0.9))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')










# Test Execution
if __name__ == '__main__':
    main()