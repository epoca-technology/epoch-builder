from typing import List
from json import load
from modules.arima_combinations import ArimaCombinations, IArimaCombinationsConfig




# CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
print("ARIMA COMBINATIONS\n")
config_file = open('ArimaCombinations_config.json')
config: IArimaCombinationsConfig = load(config_file)


# INSTANCE
# The Instance of the ArimaCombinations
arima_combinations = ArimaCombinations(config)


# BACKTEST FILES
# Once the combinations have been generated, they are stored in the output path.
arima_combinations.generate()
print(f"The files were placed in:")
print(f"{ArimaCombinations.OUTPUT_PATH}/{config['id']}_{config['focus_number']}")