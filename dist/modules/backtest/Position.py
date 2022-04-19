from typing import Tuple, List
from pandas import Series
from modules.utils import Utils
from modules.model import IPrediction
from modules.backtest import IPerformance





class Position:
    """Position Class

    Manages the positions within a Model's Backtesting as well as perform analysis on its
    retuls.

    Class Properties:
        REWARD: float
            The number of points that will be added if the position is successful
        PENALTY: float
            The number of points that will be substracted if the position is unsuccessful

    Instance Properties:
        Configuration
            take_profit: float
                The take profit percentage that will be used in positions.
            stop_loss: float
                The stop loss percentage that will be used in positions.

        Points Data:
            points: List[float]
                The history of how points have fluctuated during the process.

        Positions:
            active: Union[IPosition, None]:
                This property is populated when there is an active position. Otherwise,
                new positions can be opened.
            positions: List[IPosition]
                The list of positions that have been closed in the instance.

        Position Counters:
            successful_num: int
                The number of successful positions. Includes longs & shorts.
            long_num: int
                The number of long positions.
            successful_long_num: int
                The number of successful long positions.
            short_num: int
                The number of short positions.
            successful_short_num: int
                The number of successful short positions.
            
    """

    # The points to be added or substracted based on the position outcome
    REWARD: float = 1.0
    PENALTY: float = -1.2




    ## Initialization ##


    def __init__(self, take_profit: float, stop_loss: float):
        """Initializes the Position Instance in order to keep track of the
        Backtest's performance by model.

        Args:
            take_profit: float
                The percentage that will be used to set the take profit price on the positions.
            stop_loss: float
                The percentage that will be used to set the stop loss price on the positions.
        """
        # Init Config
        self.take_profit = take_profit
        self.stop_loss = stop_loss

        # Init Points
        self.points: List[float] = [0]

        # Init Positions
        self.active = None
        self.positions = []

        # Position Counters
        self.successful_num = 0
        self.long_num = 0
        self.successful_long_num = 0
        self.short_num = 0
        self.successful_short_num = 0






    ## Position Management ##



    def open_position(self, candlestick: Series, prediction: IPrediction) -> None:
        """Opens a new position a position based on provided params and stores it
        in the active property.

        Args:
            candlestick: Series
                The current 1m candlestic.
            prediction: IPrediction
                The prediction generated by the model.
        """
        # Calculate the exit prices
        take_profit, stop_loss = self._get_exit_prices(prediction['r'], candlestick['o'])

        # Populate the active position property
        self.active = {
            't': prediction['r'],
            'p': prediction,
            'ot': candlestick['ot'],
            'op': candlestick['o'],
            'tpp': take_profit,
            'slp': stop_loss,
        }






    def _get_exit_prices(self, position_type: int, open_price: float) -> Tuple[float, float]:
        """Calculates the take profit and the stop loss prices for a position about to be opened.

        Args:
            position_type: int
                The type of position based on the prediction. It can only be 1 or -1.
            open_price: float
                The current candlestick's open price.
        
        Returns:
            Tuple[float, float] (take_profit, stop_loss)
        """
        # Handle a long position
        if position_type == 1:
            return Utils.alter_number_by_percentage(open_price, self.take_profit), \
                Utils.alter_number_by_percentage(open_price, -(self.stop_loss))
        
        # Handle a short position
        else:
            return Utils.alter_number_by_percentage(open_price, -(self.take_profit)), \
                Utils.alter_number_by_percentage(open_price, self.stop_loss)







    def check_position(self, candlestick: Series) -> bool:
        """Checks the current candlestick against the open position. Returns
        True if the position is closed, otherwise returns False.

        Args:
            candlestick: Series
                The current 1m candlestic.
        
        Returns:
            bool
        """
        # Init the closed position status
        position_closed: bool = False

        # Check if the active position is a long and needs to be closed
        if self.active['t'] == 1:
            # Close the position if the stop loss has been hit by the low
            if candlestick['l'] <= self.active['slp']:
                self._close_position(False, self.active['slp'], candlestick['ct'])
                position_closed = True

            # Close the position if the take profit has been hit by the high
            elif candlestick['h'] >= self.active['tpp']:
                self._close_position(True, self.active['tpp'], candlestick['ct'])
                position_closed = True

        # Check if the short needs to be closed
        else:
            # Close the position if the stop loss has been hit by the high
            if candlestick['h'] >= self.active['slp']:
                self._close_position(False, self.active['slp'], candlestick['ct'])
                position_closed = True

            # Close the position if the take profit has been hit by the low
            elif candlestick['l'] <= self.active['tpp']:
                self._close_position(True, self.active['tpp'], candlestick['ct'])
                position_closed = True

        # Finally, notify if the position has been closed
        return position_closed






    def _close_position(self, outcome: bool, close_price: float, close_time: int) -> None:
        """Completes the position dict, adds it to the list, updates the points and the
        counters.

        Args:
            outcome: bool
                Wether the position was successful or not.
            close_price: float
                The price that triggered the position to be closed.
            close_time: float
                The close time of the candlestick in which the position was closed.
        """
        # Complete the position
        self.active['ct'] = close_time
        self.active['cp'] = close_price
        self.active['o'] = outcome

        # Update the points and add the new value to the position
        self.active['pts'] = self._update_points(outcome)
        
        # Update the long counters
        if self.active['t'] == 1:
            # Increment the number of longs
            self.long_num += 1

            # Handle successful counters if applies
            if outcome:
                self.successful_num += 1
                self.successful_long_num += 1

        # Update the short counters
        else:
            # Increment the number of shorts
            self.short_num += 1

            # Handle successful counters if applies
            if outcome:
                self.successful_num += 1
                self.successful_short_num += 1

        # Add the active position to the list
        self.positions.append(self.active)

        # Clear the active position
        self.active = None









    ## Points ##





    def _update_points(self, outcome: bool) -> float:
        """Updates the current points based on the provided outcome and
        returns the updated points.

        Args:
            outcome: bool
                True for successful and False for unsuccessful.

        Returns:
            float
        """
        # Calculate the outcome points
        outcome_points: float = Position.REWARD if outcome else Position.PENALTY
        
        # Add the points to the list
        self.points.append(round(self.points[-1] + outcome_points, 2))

        # Return the current points
        return self.points[-1]












    ## Performance ##





    def get_performance(self) -> IPerformance:
        """Returns the performance dictionary based on the instance
        data.

        Returns:
            IPerformance
        """
        return {
            'points': self.points[-1],
            'points_hist': self.points,
            'positions': self.positions,
            'long_num': self.long_num,
            'short_num': self.short_num,
            'long_acc': Utils.get_percentage_out_of_total(self.successful_long_num, self.long_num if self.long_num > 0 else 1),
            'short_acc': Utils.get_percentage_out_of_total(self.successful_short_num, self.short_num if self.short_num > 0 else 1),
            'general_acc': Utils.get_percentage_out_of_total(self.successful_num, len(self.positions) if len(self.positions) > 0 else 1),
        }