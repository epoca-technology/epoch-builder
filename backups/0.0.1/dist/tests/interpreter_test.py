from typing import List
import unittest
from modules.utils import Utils
from modules.model import Interpreter



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
class InterpreterTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    ## Interpreter Initialization ##



    # Can initialize a basic interpreter
    def testInitializeBasicInterpreter(self):
        i = Interpreter({"long": 1.5, "short": 1.5})
        self.assertEqual(i.long, 1.5)
        self.assertEqual(i.short, 1.5)
        self.assertFalse(i.rsi["active"])
        self.assertFalse(i.ema["active"])


    # Can initialize an interpreter with default RSI
    def testInitializeInterpreterWithDefaultRSI(self):
        i = Interpreter({"long": 0.75, "short": 1.87, "rsi": {"active": True}})
        self.assertEqual(i.long, 0.75)
        self.assertEqual(i.short, 1.87)
        self.assertTrue(i.rsi["active"])
        self.assertEqual(i.rsi["overbought"], Interpreter.DEFAULT_RSI_OVERBOUGHT)
        self.assertEqual(i.rsi["oversold"], Interpreter.DEFAULT_RSI_OVERSOLD)
        self.assertFalse(i.ema["active"])


    # Can initialize an interpreter with custom RSI
    def testInitializeInterpreterWithCustomRSI(self):
        i = Interpreter({"long": 2.78, "short": 3.85, "rsi": {"active": True, "overbought": 70.0, "oversold": 30.0}})
        self.assertEqual(i.long, 2.78)
        self.assertEqual(i.short, 3.85)
        self.assertTrue(i.rsi["active"])
        self.assertEqual(i.rsi["overbought"], 70.0)
        self.assertEqual(i.rsi["oversold"], 30.0)
        self.assertFalse(i.ema["active"])


    # Can initialize an interpreter with default EMA
    def testInitializeInterpreterWithDefaultEMA(self):
        i = Interpreter({"long": 0.75, "short": 1.87, "ema": {"active": True}})
        self.assertEqual(i.long, 0.75)
        self.assertEqual(i.short, 1.87)
        self.assertFalse(i.rsi["active"])
        self.assertTrue(i.ema["active"])
        self.assertEqual(i.ema["distance"], Interpreter.DEFAULT_EMA_DISTANCE)


    # Can initialize an interpreter with custom EMA
    def testInitializeInterpreterWithCustomEMA(self):
        i = Interpreter({"long": 2.78, "short": 3.85, "ema": {"active": True, "distance": 1.5}})
        self.assertEqual(i.long, 2.78)
        self.assertEqual(i.short, 3.85)
        self.assertFalse(i.rsi["active"])
        self.assertTrue(i.ema["active"])
        self.assertEqual(i.ema["distance"], 1.5)


    # Can initialize an interpreter with default RSI & EMA
    def testInitializeInterpreterWithDefaultRSIAndEMA(self):
        i = Interpreter({"long": 1, "short": 1, "rsi": {"active": True}, "ema": {"active": True}})
        self.assertTrue(i.rsi["active"])
        self.assertEqual(i.rsi["overbought"], Interpreter.DEFAULT_RSI_OVERBOUGHT)
        self.assertEqual(i.rsi["oversold"], Interpreter.DEFAULT_RSI_OVERSOLD)
        self.assertTrue(i.ema["active"])
        self.assertEqual(i.ema["distance"], Interpreter.DEFAULT_EMA_DISTANCE)


    # Can initialize an interpreter with custom RSI & EMA
    def testInitializeInterpreterWithCustomRSIAndEMA(self):
        i = Interpreter({
            "long": 1, 
            "short": 1, 
            "rsi": {"active": True, "overbought": 60, "oversold": 40}, 
            "ema": {"active": True, "distance": 2}}
        )
        self.assertTrue(i.rsi["active"])
        self.assertEqual(i.rsi["overbought"], 60.0)
        self.assertEqual(i.rsi["oversold"], 40.0)
        self.assertTrue(i.ema["active"])
        self.assertEqual(i.ema["distance"], 2.0)








    ## Predictions Interpretation ##



    # Can interpret a basic long
    def testBasicLongInterpretation(self):
        # Init the interpreter
        i = Interpreter({"long": 0.5, "short": 0.5})

        # Can interpret a long position if the prediction change is equals
        result, description = i.get_interpretation(_get_preds(0.5), None, None, None)
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # Can interpret a long position if the prediction change is greater
        result, description = i.get_interpretation(_get_preds(2.5), None, None, None)
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.get_interpretation(_get_preds(0.4), None, None, None)
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')




    # Can interpret a basic short
    def testBasicShortInterpretation(self):
        # Init the interpreter
        i = Interpreter({"long": 0.5, "short": 0.5})

        # Can interpret a long position if the prediction change is equals
        result, description = i.get_interpretation(_get_preds(-0.5), None, None, None)
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # Can interpret a long position if the prediction change is greater
        result, description = i.get_interpretation(_get_preds(-2.5), None, None, None)
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.get_interpretation(_get_preds(-0.4), None, None, None)
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')





    # Can interpret a long w/ RSI
    def testLongWithRSIInterpretation(self):
        # Init the interpreter
        i = Interpreter({"long": 0.5, "short": 0.5, "rsi": {"active": True, "overbought": 70, "oversold": 30}})

        # Init the predictions
        preds = _get_preds(8.75)

        # Can interpret a long position if the RSI is not overbought
        result, description = i.get_interpretation(preds, 69.99, None, None)
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # If the RSI is overbought, the long is neutralized
        result, description = i.get_interpretation(preds, 70, None, None)
        self.assertEqual(result, 0)
        self.assertEqual(description, 'long-neutralized-by-rsi-overbought')

        # If the RSI is overbought, the long is neutralized
        result, description = i.get_interpretation(preds, 89, None, None)
        self.assertEqual(result, 0)
        self.assertEqual(description, 'long-neutralized-by-rsi-overbought')




    # Can interpret a short w/ RSI
    def testShortWithRSIInterpretation(self):
        # Init the interpreter
        i = Interpreter({"long": 0.5, "short": 0.5, "rsi": {"active": True, "overbought": 70, "oversold": 30}})

        # Init the predictions
        preds = _get_preds(-2.55)

        # Can interpret a short position if the RSI is not oversold
        result, description = i.get_interpretation(preds, 31, None, None)
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # If the RSI is oversold, the short is neutralized
        result, description = i.get_interpretation(preds, 30, None, None)
        self.assertEqual(result, 0)
        self.assertEqual(description, 'short-neutralized-by-rsi-oversold')

        # If the RSI is oversold, the long is neutralized
        result, description = i.get_interpretation(preds, 11.65, None, None)
        self.assertEqual(result, 0)
        self.assertEqual(description, 'short-neutralized-by-rsi-oversold')






    # Can interpret a long w/ EMA
    def testLongWithEMAInterpretation(self):
        # Init the interpreter
        i = Interpreter({"long": 0.5, "short": 0.5, "ema": {"active": True, "distance": 0.5}})

        # Init the predictions
        preds = _get_preds(4.33)

        # Can interpret a long position if the EMA is in an uptrend
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, -0.6))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # Can interpret a long position if the EMA is not in a downtrend
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, -0.3))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # If the EMA is in a downtrend, the long is neutralized
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, 0.5))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'long-neutralized-by-ema-downtrend')

        # If the EMA is in a downtrend, the long is neutralized
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, 1.5))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'long-neutralized-by-ema-downtrend')




    # Can interpret a short w/ EMA
    def testShortWithEMAInterpretation(self):
        # Init the interpreter
        i = Interpreter({"long": 0.5, "short": 0.5, "ema": {"active": True, "distance": 0.5}})

        # Init the predictions
        preds = _get_preds(-8.58)

        # Can interpret a short position if the EMA is in a downtrend
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, 0.6))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # Can interpret a short position if the EMA is not in an uptrend
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, -0.1))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # If the EMA is in an uptrend, the short is neutralized
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, -0.5))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'short-neutralized-by-ema-uptrend')

        # If the EMA is in an uptrend, the short is neutralized
        result, description = i.get_interpretation(preds, None, BASE, Utils.alter_number_by_percentage(BASE, -1.5))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'short-neutralized-by-ema-uptrend')






    # Can interpret longs with RSI and EMA
    def testLongWithRSIAndEMAInterpretation(self):
        # Init the interpreter
        i = Interpreter({
            "long": 0.5, 
            "short": 0.5, 
            "rsi": {"active": True, "overbought": 70, "oversold": 30}, 
            "ema": {"active": True, "distance": 0.5}}
        )

        # Init the Prediction Lists
        preds = _get_preds(2.55)

        # Can interpret a long position if the RSI is not overbought and the EMA is not in a downtrend
        result, description = i.get_interpretation(preds, 69.99, BASE, Utils.alter_number_by_percentage(BASE, 0.1))
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # A long is neutralized if the RSI is overbought
        result, description = i.get_interpretation(preds, 85.46, BASE, Utils.alter_number_by_percentage(BASE, 0.7))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'long-neutralized-by-rsi-overbought')

        # A long is neutralized if the EMA is in a downtrend
        result, description = i.get_interpretation(preds, 58.85, BASE, Utils.alter_number_by_percentage(BASE, 0.7))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'long-neutralized-by-ema-downtrend')





    # Can interpret shorts with RSI and EMA
    def testShortWithRSIAndEMAInterpretation(self):
        # Init the interpreter
        i = Interpreter({
            "long": 0.5, 
            "short": 0.5, 
            "rsi": {"active": True, "overbought": 70, "oversold": 30}, 
            "ema": {"active": True, "distance": 0.5}}
        )

        # Init the Prediction Lists
        preds = _get_preds(-3.16)

        # Can interpret a short position if the RSI is not oversold and the EMA is not in an uptrend
        result, description = i.get_interpretation(preds, 30.1, BASE, Utils.alter_number_by_percentage(BASE, 0.1))
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # A short is neutralized if the RSI is oversold
        result, description = i.get_interpretation(preds, 16.46, BASE, Utils.alter_number_by_percentage(BASE, -0.6))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'short-neutralized-by-rsi-oversold')

        # A short is neutralized if the EMA is in an uptrend
        result, description = i.get_interpretation(preds, 47.1, BASE, Utils.alter_number_by_percentage(BASE, -0.6))
        self.assertEqual(result, 0)
        self.assertEqual(description, 'short-neutralized-by-ema-uptrend')





# Test Execution
if __name__ == '__main__':
    unittest.main()