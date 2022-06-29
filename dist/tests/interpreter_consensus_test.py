import unittest
from modules.database.Database import Database
from modules.interpreter.ConsensusInterpreter import ConsensusInterpreter




## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")


    


# Test Class
class ConsensusInterpreterTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    ## Interpreter Initialization ##



    # Can initialize an interpreter
    def testInitializeInterpreter(self):
        i = ConsensusInterpreter({"min_consensus": 2})
        self.assertEqual(i.min_consensus, 2)








    ## Predictions Interpretation ##

    # Can interpret 2-out-of-2
    def canInterpreterTwoOutOfTwo(self):
        # Init the interpreter
        i = ConsensusInterpreter({"min_consensus": 2})

        ## Longs ##
        result, description = i.interpret([1, 1])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        ## Shorts ##
        result, description = i.interpret([-1, -1])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        ## Neutrals ##
        result, description = i.interpret([1, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([-1, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([0, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([0, 1])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([0, -1])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')




    # Can interpret 2-out-of-3
    def canInterpreterTwoOutOfThree(self):
        # Init the interpreter
        i = ConsensusInterpreter({"min_consensus": 2})

        ## Longs ##
        result, description = i.interpret([1, 1, 0])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        result, description = i.interpret([1, 1, -1])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        result, description = i.interpret([1, 1, 1])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')


        ## Shorts ##
        result, description = i.interpret([-1, -1, 0])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        result, description = i.interpret([-1, -1, 1])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')

        result, description = i.interpret([-1, -1, -1])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')


        ## Neutrals ##
        result, description = i.interpret([-1, 1, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([-1, 0, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([0, 1, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([0, 1, -1])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')




    # Can interpret 3-out-of-3
    def canInterpreterThreeOutOfThree(self):
        # Init the interpreter
        i = ConsensusInterpreter({"min_consensus": 3})

        ## Longs ##
        result, description = i.interpret([1, 1, 1])
        self.assertEqual(result, 1)
        self.assertEqual(description, 'long')

        ## Shorts ##
        result, description = i.interpret([-1, -1, -1])
        self.assertEqual(result, -1)
        self.assertEqual(description, 'short')


        ## Neutrals ##
        result, description = i.interpret([1, 1, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([1, 1, -1])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([-1, -1, 0])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([-1, -1, 1])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')

        result, description = i.interpret([1, -1, -1])
        self.assertEqual(result, 0)
        self.assertEqual(description, 'neutral')



    # Can interpret 3-out-of-5
    def canInterpreterThreeOutOfFive(self):
        # Init the interpreter
        i = ConsensusInterpreter({"min_consensus": 3})

        ## Longs ##
        result, description = i.interpret([1, 1, 1, 0, 0])
        self.assertEqual(result, 1)

        result, description = i.interpret([1, 1, 1, -1, 0])
        self.assertEqual(result, 1)

        result, description = i.interpret([1, 1, 1, -1, -1])
        self.assertEqual(result, 1)

        result, description = i.interpret([1, 1, 1, 1, 0])
        self.assertEqual(result, 1)

        result, description = i.interpret([1, 1, 1, 1, -1])
        self.assertEqual(result, 1)

        result, description = i.interpret([1, 1, 1, 1, 1])
        self.assertEqual(result, 1)

        ## Shorts ##
        result, description = i.interpret([-1, -1, -1, 0, 0])
        self.assertEqual(result, -1)

        result, description = i.interpret([-1, -1, -1, 1, 0])
        self.assertEqual(result, -1)

        result, description = i.interpret([-1, -1, -1, 1, 1])
        self.assertEqual(result, -1)

        result, description = i.interpret([-1, -1, -1, -1, 0])
        self.assertEqual(result, -1)

        result, description = i.interpret([-1, -1, -1, -1, 1])
        self.assertEqual(result, -1)

        result, description = i.interpret([-1, -1, -1, -1, -1])
        self.assertEqual(result, -1)

        ## Neutrals ##
        result, description = i.interpret([1, 1, 0, 0, 0])
        self.assertEqual(result, 0)

        result, description = i.interpret([1, 1, -1, -1, 0])
        self.assertEqual(result, 0)

        result, description = i.interpret([1, 1, -1, 0, -1])
        self.assertEqual(result, 0)

        result, description = i.interpret([0, 0, 0, 0, 0])
        self.assertEqual(result, 0)



    # Can interpret 5-out-of-5
    def canInterpreterFiveOutOfFive(self):
        # Init the interpreter
        i = ConsensusInterpreter({"min_consensus": 5})

        ## Longs ##
        result, description = i.interpret([1, 1, 1, 1, 1])
        self.assertEqual(result, 1)

        ## Shorts ##
        result, description = i.interpret([-1, -1, -1, -1, -1])
        self.assertEqual(result, -1)

        ## Neutrals ##
        result, description = i.interpret([1, 1, 1, 1, 0])
        self.assertEqual(result, 0)

        result, description = i.interpret([1, 1, 1, 1, -1])
        self.assertEqual(result, 0)

        result, description = i.interpret([-1, -1, -1, -1, 0])
        self.assertEqual(result, 0)

        result, description = i.interpret([-1, -1, -1, -1, 1])
        self.assertEqual(result, 0)

        result, description = i.interpret([0, 0, 0, 0, 0])
        self.assertEqual(result, 0)






# Test Execution
if __name__ == '__main__':
    unittest.main()