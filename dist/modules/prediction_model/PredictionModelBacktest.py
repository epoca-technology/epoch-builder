from typing import List, Union, Tuple
from modules._types import IPredictionResult, IPrediction, IBacktestPositionType, IBacktestPosition, IBacktestPerformance,\
    ILookbackIndexer, ICandlestick, IPredictionStateIntensity
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch





class PredictionModelBacktest:
    """PredictionModelBacktest Class

    This class builds handles the backtesting of Prediction Models.

    Instance Properties:
        features_num: int
            The total number of features per regression.
        lookback_indexer: ILookbackIndexer
            The indexer used to relate 1m candlesticks with test dataset indexes.
        initial_balance: float
            The balance the model has prior to trading.
        equity_size: float
            The total amount of money that is actually traded on each position.
        open_fee: float
            The cost of opening a position.
        price_change_requirement: float
            The percentage that will be used to calculate the position exit combinations.
        gross_profit: float
        gross_loss: float
            The gross profit or loss of a position.
        successful_close_fee: float
        unsuccessful_close_fee: float
            The fee charged when a position is closed.
        net_successful_fee: float
        net_unsuccessful_fee: float
            The net fee to be charged based on the position outcome.
        net_profit: float
        net_loss: float
            The total amount of money that will be added or deducted from the balance
            on position close based on the outcome.
        current_balance: float
            The current balance of the simulation. It is updated every time a position is
            closed.
        fees: float
            Accumulated exchange fees during the backtest. This value includes both, open
            and close fees.
        active: Union[IBacktestPosition, None]
            The position that is currently active and needs to be checked on new candlesticks.
        positions: List[IBacktestPosition]
            The list of closed positions.
        increase_num: int
        increase_successful_num: int
        decrease_num: int
        decrease_successful_num: int
            Position counters.
        min_increase_sum: float
        min_decrease_sum: float
            The minimum increase and decrease sums required to generate
            non-neutral predictions.
    """





    ####################
    ## Initialization ##
    ####################




    def __init__(self, features_num: int, lookback_indexer: ILookbackIndexer):
        """Initializes the PredictionModelBacktest Instance.

        Args:
            features_num: int
                The total number of features per regression.
            lookback_indexer: ILookbackIndexer
                The 1m candlestick indexer.
        """
        # Initialize the number of features
        self.features_num: int = features_num

        # Initialize the lookback indexer
        self.lookback_indexer: ILookbackIndexer = lookback_indexer

        # Calculate the initial balance
        self.initial_balance: float = round(Epoch.POSITION_SIZE * 1.5, 2)

        # Calculate the equity size by trade
        self.equity_size: float = round(Epoch.POSITION_SIZE * Epoch.LEVERAGE, 2)

        # Calculate the fee for opening a trade
        self.open_fee: float = (self.equity_size / 100) * Epoch.EXCHANGE_FEE













    ##########################
    ## Backtest Performance ##
    ##########################




    def calculate_performance(
        self, 
        price_change_requirement: float,
        min_increase_sum: float,
        min_decrease_sum: float,
        features: List[List[float]], 
        features_sum: List[float]
    ) -> IBacktestPerformance:
        """Calculates the backtest performance for a given model.

        Args:
            price_change_requirement: float
                The percentage used to calculate the position exit 
                combinations (Take Profit / Stop Loss).
            min_increase_sum: float
            min_decrease_sum: float
                The minimum increase and decrease sums required to generate
                non-neutral predictions.
            features: List[List[float]]
                The raw list of features that will be used to build the 
                prediction dict that will be attached to positions.
            features_sum: List[float]
                The list of features sums that will be used to quickly identify
                non-neutral predictions.

        Returns:
            IBacktestPerformance
        """
        # Init the price change requirement
        self.price_change_requirement: float = price_change_requirement

        # Calculate the gross profit|loss per trade
        self.gross_profit: float = Utils.alter_number_by_percentage(self.equity_size, self.price_change_requirement) - self.equity_size
        self.gross_loss: float = self.equity_size - Utils.alter_number_by_percentage(self.equity_size, -self.price_change_requirement)

        # Calculate the close fee for each scenario
        self.successful_close_fee: float = (Utils.alter_number_by_percentage(self.equity_size, self.price_change_requirement) / 100) * Epoch.EXCHANGE_FEE
        self.unsuccessful_close_fee: float = (Utils.alter_number_by_percentage(self.equity_size, -self.price_change_requirement) / 100) * Epoch.EXCHANGE_FEE

        # Calculate the fee per outcome
        self.net_successful_fee: float = self.open_fee + self.successful_close_fee
        self.net_unsuccessful_fee: float = self.open_fee + self.unsuccessful_close_fee

        # Calculate the net profit|loss per trade
        self.net_profit: float = self.gross_profit - self.net_successful_fee
        self.net_loss: float = self.gross_loss + self.net_unsuccessful_fee

        # Position specific properties
        self.current_balance: float = self.initial_balance
        self.fees: float = 0
        self.active: Union[IBacktestPosition, None] = None
        self.positions: List[IBacktestPosition] = []
        self.increase_num: int = 0
        self.increase_successful_num: int = 0
        self.decrease_num: int = 0
        self.decrease_successful_num: int = 0

        # Minimum increase and decrease sums
        self.min_increase_sum: float = min_increase_sum
        self.min_decrease_sum: float = min_decrease_sum

        # Idle Until
        # The model will remain in an idle state until a candlestick's ot is greater than this value.
        idle_until: int = 0

        # Iterate for as long there are features and enough balance to cover the position size
        for candlestick in Candlestick.DF.to_records():
            # Init the current index
            current_index: int = self.lookback_indexer[str(candlestick["ot"])]

            # Make sure there are features and sufficient balance to cover the position size
            if current_index < self.features_num and self.current_balance >= Epoch.POSITION_SIZE:
                # Check if there is an active position
                if self.active is not None:
                    # Check the position against the new candlestick
                    closed_position: bool = self._check_position(candlestick)

                    # If the position has been closed, enable the idle state
                    if closed_position:
                        idle_until = Utils.add_minutes(candlestick["ct"], Epoch.IDLE_MINUTES_ON_POSITION_CLOSE)

                # Otherwise, check if a position can be opened
                elif (self.active == None) and (candlestick["ot"] >= idle_until):
                    # Retrieve the prediction result as long as there is at least 5 sums in the past
                    pred_result: IPredictionResult = 0
                    if current_index > 5:
                        pred_result = self._get_prediction_result(features_sum[current_index-5:current_index+1])

                    # If the result isn't neutral, open a position
                    if pred_result != 0:
                        self._open_position(candlestick, {
                            "r": pred_result,
                            "t": int(candlestick["ot"]),
                            "f": features[current_index]
                        })

            # Otherwise, stop the process
            else:
                break

        # Finally, return the performance
        return self._build_performance()






    def _get_prediction_result(self, sums: List[float]) -> IPredictionResult:
        """Retrieves a prediction result based on the sum of all the features
        at the current index as well as the general trend of the sums.

        Args:
            sums: List[float]
                The list of sums ordered ascendingly where the current
                sum is the last.

        Returns:
            IPredictionResult
        """
        # Calculate the prediction trend
        increasing, increasing_strongly, decreasing, decreasing_strongly, intensity = \
            self._calculate_prediction_trend(sums)

        # If the feature sum meets the requirement and the trend is increasing, open a long
        if  (sums[-1] >= self.min_increase_sum and increasing and intensity >= 1) or \
            (sums[-1] <= self.min_decrease_sum and increasing_strongly and intensity >= 2):
            return 1

        # If the feature sum meets the requirement and the trend is decreasing, open a short
        elif (sums[-1] <= self.min_decrease_sum and decreasing and intensity <= -1) or \
             (sums[-1] >= self.min_increase_sum and decreasing_strongly and intensity <= -2):
            return -1

        # Otherwise, the model is neutral
        else:
            return 0




    def _calculate_prediction_trend(self, sums: List[float]) -> Tuple[bool, bool, bool, bool, IPredictionStateIntensity]:
        """Determines if the prediction sums are increasing or decreasing based
        on the current and the 5 previous items. It also calculates the intensity
        of the direction.

        Args:
            sums: List[float]

        Returns:
            Tuple[bool, bool, bool, bool, IPredictionStateIntensity]
            increasing, increasing_strongly, decreasing, decreasing_strongly, intensity
        """
        # Check if the trend is increasing or decreasing
        increasing: bool = sums[-1] > sums[-2] and sums[-2] > sums[-3] and sums[-3] > sums[-4]
        increasing_strongly: bool = sums[-1] > sums[-2] and sums[-2] > sums[-3] and sums[-3] > sums[-4] and sums[-4] > sums[-5] and sums[-5] > sums[-6]
        decreasing: bool = sums[-1] < sums[-2] and sums[-2] < sums[-3] and sums[-3] < sums[-4]
        decreasing_strongly: bool = sums[-1] < sums[-2] and sums[-2] < sums[-3] and sums[-3] < sums[-4] and sums[-4] < sums[-5] and sums[-5] < sums[-6]

        # Calculate the intensity of the direction
        intensity: IPredictionStateIntensity = self._calculate_state_intensity(sums[0], sums[-1])

        # Finally, pack the results and return them
        return increasing, increasing_strongly, decreasing, decreasing_strongly, intensity
        



    def _calculate_state_intensity(self, initial_sum: float, current_sum: float) -> IPredictionStateIntensity:
        """Based on the initial and current sum, it calculates the intensity
        of the trend's direction.

        Args:
            initial_sum: float
                The trend sum from 5 candlesticks ago.
            current_sum: float
                The trend sum in the current candlestick.

        Returns:
            IPredictionStateIntensity
        """
        # Handle a positive sum
        if initial_sum > 0:
            # Check if there has been an increase
            if initial_sum < current_sum:
                if current_sum >= Utils.alter_number_by_percentage(initial_sum, 8):
                    return 2
                elif current_sum >= Utils.alter_number_by_percentage(initial_sum, 4):
                    return 1
                else:
                    return 0

            # Check if there has been a decrease
            elif initial_sum > current_sum:
                if current_sum <= Utils.alter_number_by_percentage(initial_sum, -8):
                    return -2
                elif current_sum <= Utils.alter_number_by_percentage(initial_sum, -4):
                    return -1
                else:
                    return 0

            # If the initial sum is equals to the current one, there is no intensity
            else:
                return 0

        # Handle a negative sum
        elif initial_sum < 0:
            # Check if there has been a decrease
            if initial_sum > current_sum:
                if current_sum <= Utils.alter_number_by_percentage(initial_sum, 8):
                    return -2
                elif current_sum <= Utils.alter_number_by_percentage(initial_sum, 4):
                    return -1
                else:
                    return 0

            # Check if there has been an increase
            if initial_sum < current_sum:
                if current_sum >= Utils.alter_number_by_percentage(initial_sum, -8):
                    return 2
                elif current_sum >= Utils.alter_number_by_percentage(initial_sum, -4):
                    return 1
                else:
                    return 0

            # If the initial sum is equals to the current one, there is no intensity
            else:
                return 0

        # If the initial sum is equals to 0, there is no intensity
        else:
            return 0






    def _build_performance(self) -> IBacktestPerformance:
        """Once the backtest is completed, it builds the performance dict
        containing all the details.

        Returns:
            IBacktestPerformance
        """ 
        # Init values
        total_positions = self.increase_num + self.decrease_num
        
        # Return the performance build
        return {
            # General
            "position_size": Epoch.POSITION_SIZE,
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "profit": self.current_balance - self.initial_balance,
            "fees": self.fees,
            "leverage": Epoch.LEVERAGE,
            "exchange_fee": Epoch.EXCHANGE_FEE,
            "idle_minutes_on_position_close": Epoch.IDLE_MINUTES_ON_POSITION_CLOSE,

            # Positions
            "positions": self.positions,
            "increase_num": self.increase_num,
            "decrease_num": self.decrease_num,
            "increase_outcome_num": self.increase_successful_num + (self.decrease_num - self.decrease_successful_num),
            "decrease_outcome_num": self.decrease_successful_num + (self.increase_num - self.increase_successful_num),

            # Accuracy
            "increase_accuracy": Utils.get_percentage_out_of_total(self.increase_successful_num, self.increase_num if self.increase_num > 0 else 1),
            "decrease_accuracy": Utils.get_percentage_out_of_total(self.decrease_successful_num, self.decrease_num if self.decrease_num > 0 else 1),
            "accuracy": Utils.get_percentage_out_of_total(self.increase_successful_num + self.decrease_successful_num, total_positions if total_positions > 0 else 1)
        }









    #########################
    ## Position Management ##
    #########################




    def _open_position(self, candlestick: ICandlestick, prediction: IPrediction) -> None:
        """Opens a position based on the current candlestick and the prediction.

        Args:
            candlestick: ICandlestick
                The current 1m candlestick.
            prediction: IBacktestPositionType
                The non-neutral prediction generated by the model.
        """
        # Calculate exit prices
        take_profit_price, stop_loss_price = self._get_exit_prices(prediction["r"], candlestick["o"])

        # Open the position
        self.active = {
            "t": prediction["r"],
            "p": prediction,
            "ot": prediction["t"],
            "op": candlestick["o"],
            "tpp": take_profit_price,
            "slp": stop_loss_price
        }





    def _get_exit_prices(self, position_type: IBacktestPositionType, open_price: float) -> Tuple[float, float]:
        """Calculates the take profit and the stop loss for a new position.

        Args:
            position_type: int
                The prediction generated by the model (long or short).
            open_price: float
                The price in which the position will be opened at.

        Returns:
            Tuple[float, float]
            (take_profit, stop_loss)
        """
        # Handle a long position
        if position_type == 1:
            return Utils.alter_number_by_percentage(open_price, self.price_change_requirement), \
                Utils.alter_number_by_percentage(open_price, -(self.price_change_requirement))
        
        # Handle a short position
        else:
            return Utils.alter_number_by_percentage(open_price, -(self.price_change_requirement)), \
                Utils.alter_number_by_percentage(open_price, self.price_change_requirement)







    def _check_position(self, candlestick: ICandlestick) -> bool:
        """Checks the active position against a new candlestick. Returns
        True if the position was closed.

        Args:
            candlestick: ICandlestick
                The current 1m candlestick.
        
        Returns:
            bool
        """
        # Init the closed position status
        position_closed: bool = False

        # Check if the active position is a long and needs to be closed
        if self.active["t"] == 1:
            # Close the position if the stop loss has been hit by the low
            if candlestick["l"] <= self.active["slp"]:
                self._close_position(False, self.active["slp"], candlestick["ct"])
                position_closed = True

            # Close the position if the take profit has been hit by the high
            elif candlestick["h"] >= self.active["tpp"]:
                self._close_position(True, self.active["tpp"], candlestick["ct"])
                position_closed = True

        # Check if the short needs to be closed
        else:
            # Close the position if the stop loss has been hit by the high
            if candlestick["h"] >= self.active["slp"]:
                self._close_position(False, self.active["slp"], candlestick["ct"])
                position_closed = True

            # Close the position if the take profit has been hit by the low
            elif candlestick["l"] <= self.active["tpp"]:
                self._close_position(True, self.active["tpp"], candlestick["ct"])
                position_closed = True

        # Finally, notify if the position has been closed
        return position_closed







    def _close_position(self, outcome: bool, close_price: float, close_time: int) -> None:
        """Closes an active position based on the provided outcome.

        Args:
            outcome: bool
                The final outcome of the position. True stands for successful.
            close_price: float
                The price in which the position was closed.
            close_time: int
                The time in which the position was closed.

        Returns:
            None
        """
        # Complete the position
        self.active["ct"] = int(close_time)
        self.active["cp"] = close_price
        self.active["o"] = outcome
        self.active["b"] = self._update_balance(outcome)
        
        # Update the long counters
        if self.active["t"] == 1:
            # Increment the number of longs
            self.increase_num += 1

            # Handle successful counters if applies
            if outcome:
                self.increase_successful_num += 1

        # Update the short counters
        else:
            # Increment the number of shorts
            self.decrease_num += 1

            # Handle successful counters if applies
            if outcome:
                self.decrease_successful_num += 1

        # Add the active position to the list
        self.positions.append(self.active)

        # Clear the active position
        self.active = None






    def _update_balance(self, outcome: bool) -> float:
        """Updates the current balance and retrieves the new one based
        on the position outcome.

        Args:
            outcome: bool
                The outcome of the position. True means successful.

        Returns:
            float
        """
        # Calculate the outcome value and update the current balance
        self.current_balance += self.net_profit if outcome else -(self.net_loss)

        # Calculate the fee and update the accumulator
        self.fees += self.net_successful_fee if outcome else self.net_unsuccessful_fee
        
        # Return the new value
        return self.current_balance







    

    ######################
    ## Balance Drawdown ##
    ######################




    @staticmethod
    def calculate_largest_balance_drawdown(initial_balance: float, positions: List[IBacktestPosition]) -> float:
        """Calculates the largest balance drawdown the model experienced during the backtest process.

        Args:
            initial_balance: float
                The balance when the backtest process started.
            positions: List[IBacktestPosition]
                The list of positions executed during the process.

        Returns:
            float
        """
        # Initialize the balance history
        balance_hist: List[float] = [initial_balance] + [pos["b"] for pos in positions]

        # Ensure at least 1 position was executed
        txs: int = len(balance_hist)
        if txs > 1:
            # Initialize the drawdowns list
            drawdowns: List[float] = []

            # Iterate over each element
            for i in range(txs):
                # Ensure there are enough items
                if i < (txs - 1):
                    # Calculate the smallest value after the current index
                    smallest: float = min(balance_hist[i+1:])

                    # Calculate the size and add it to the list
                    drawdowns.append(Utils.get_percentage_change(balance_hist[i], smallest))

            # Finally, return the largest drawdown
            return min(drawdowns)

        # Otherwise, there is no drawdown
        else:
            return 0