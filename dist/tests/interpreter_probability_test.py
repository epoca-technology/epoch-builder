import unittest
from modules.interpreter import ProbabilityInterpreter





# Test Class
class ProbabilityInterpreterTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    ## Interpreter Initialization ##



    # Can initialize an interpreter
    def testInitializeInterpreter(self):
        i = ProbabilityInterpreter({"min_probability": 0.55})
        self.assertEqual(i.min_probability, 0.55)



    # Cannot initialize an interpreter with invalid configuration
    def testInitializeInterpreterWithInvalidConfig(self):
        with self.assertRaises(ValueError):
            ProbabilityInterpreter({"min_probability": 0.4})
        with self.assertRaises(ValueError):
            ProbabilityInterpreter({"min_probability": 1})





    ## Predictions Interpretation ##



    # Can interpret a basic long
    def testBasicLongInterpretation(self):
        # Init the interpreter
        i = ProbabilityInterpreter({"min_probability": 0.55})

        # Can interpret a long position if the probability is equals
        result, description = i.interpret([0.55, 0.45])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # Can interpret a long position if the probability is greater
        result, description = i.interpret([0.56853463, 0.43146537])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        # If the probability does not meet the requirement, it returns neutral
        result, description = i.interpret([0.5, 0.5])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')




    # Can interpret a basic short
    def testBasicShortInterpretation(self):
        # Init the interpreter
        i = ProbabilityInterpreter({"min_probability": 0.65})

        # Can interpret a short position if the probability is equals
        result, description = i.interpret([0.35, 0.65])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # Can interpret a short position if the probability is greater
        result, description = i.interpret([0.30821879, 0.69178121])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        # If the change does not meet the requirement, it returns neutral
        result, description = i.interpret([0.36, 0.64])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')






    # Cannot interpret if invalid data is provided
    def testInterpretWithInvalidData(self):
        # Init the interpreter
        i = ProbabilityInterpreter({"min_probability": 0.55})
        with self.assertRaises(ValueError):
            i.interpret([1])
        with self.assertRaises(ValueError):
            i.interpret([1, 2, 3, 4])






# Test Execution
if __name__ == '__main__':
    unittest.main()