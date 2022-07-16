from typing import Union
from modules.types import IBacktestConfig





# Backtest Configuration Unit Test
# This configuration will run the Backtest process on each model type and output the results.
def _backtest_config_ut(price_change_requirement: float, idle_minutes_on_position_close: int) -> IBacktestConfig:
    return {
        "id": "UNIT_TEST",
        "description": "The purpose of this test is to make sure the Backtest Module can run any Model.",
        "take_profit": price_change_requirement,
        "stop_loss": price_change_requirement,
        "idle_minutes_on_position_close": idle_minutes_on_position_close,
        "models": [
            {
                "id": "A212",
                "arima_models": [{"lookback": 150,"predictions": 10,"arima": { "p": 2, "d": 1, "q": 2 },"interpreter": { "long": 0.05, "short": 0.05 }}]
            },
            {
                "id": "R_UNIT_TEST",
                "regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": {"long": 1,"short": 1}}]
            },
            {
                "id": "C_UNIT_TEST",
                "classification_models": [{"classification_id": "C_UNIT_TEST", "interpreter": {"min_probability": 0.6}}]
            },
            {
                "id": "CON_UNIT_TEST",    
                "arima_models": [{"lookback": 150,"predictions": 10,"arima": { "p": 2, "d": 1, "q": 2 },"interpreter": { "long": 0.05, "short": 0.05 }}],
                "regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": { "long": 1, "short": 1 }}],
                "classification_models": [{"classification_id": "C_UNIT_TEST", "interpreter": { "min_probability": 0.6 }}],
                "consensus_model": { "interpreter": { "min_consensus": 2 } }
            }
        ]
    }