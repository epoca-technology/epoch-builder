from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch



# EPOCH MANAGEMENT
# Args:
#   --action "create"|"classification_training_data_ut"|"classification_training_data"|"export"
# 
# Create Args:
#   --id? "_ALPHA" -> Mandatory arg when creating an epoch
#   --epoch_width? "24"
#   --seed? "60184"
#   --train_split? "0.85"
#   --backtest_split? "0.3"
#   --regression_lookback? "128"
#   --regression_predictions? "32"
#   --model_discovery_steps? "5"
#   --idle_minutes_on_position_close? "30"
#
# Classification Training Data Args:
#   --training_data_file_name? "ea998c4d-9142-435f-8cc9-c5804ed5c1e8.json" -> Mandatory arg when setting Class. Training Data IDs
#
# Export Args:
#   ...
endpoint_name: str = "EPOCH MANAGEMENT"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--action", dest="action")

# Create specific args
parser.add_argument("--id", dest="id", nargs='?')
parser.add_argument("--epoch_width", dest="epoch_width", nargs='?')
parser.add_argument("--seed", dest="seed", nargs='?')
parser.add_argument("--train_split", dest="train_split", nargs='?')
parser.add_argument("--backtest_split", dest="backtest_split", nargs='?')
parser.add_argument("--regression_lookback", dest="regression_lookback", nargs='?')
parser.add_argument("--regression_predictions", dest="regression_predictions", nargs='?')
parser.add_argument("--model_discovery_steps", dest="model_discovery_steps", nargs='?')
parser.add_argument("--idle_minutes_on_position_close", dest="idle_minutes_on_position_close", nargs='?')

# Classification Training Data specific args
parser.add_argument("--training_data_file_name", dest="training_data_file_name", nargs='?')
args = parser.parse_args()



# Create Epoch
# Creates a brand new Epoch with all the required directories and files.
if args.action == "create":
    # Init the values or fill them with defaults
    epoch_width: int = int(args.epoch_width) if args.epoch_width.isdigit() else Epoch.DEFAULTS["epoch_width"]
    seed: int = int(args.seed) if args.seed.isdigit() else Epoch.DEFAULTS["seed"]
    train_split: float = float(args.train_split) if args.train_split.isdigit() else Epoch.DEFAULTS["train_split"]
    backtest_split: float = float(args.backtest_split) if args.backtest_split.isdigit() else Epoch.DEFAULTS["backtest_split"]
    regression_lookback: int = int(args.regression_lookback) if args.regression_lookback.isdigit() else Epoch.DEFAULTS["regression_lookback"]
    regression_predictions: int = int(args.regression_predictions) if args.regression_predictions.isdigit() else Epoch.DEFAULTS["regression_predictions"]
    model_discovery_steps: int = int(args.model_discovery_steps) if args.model_discovery_steps.isdigit() else Epoch.DEFAULTS["model_discovery_steps"]
    idle_minutes_on_position_close: int = int(args.idle_minutes_on_position_close) if args.idle_minutes_on_position_close.isdigit() else Epoch.DEFAULTS["idle_minutes_on_position_close"]

    # Finally, create the epoch
    Epoch.create(
        id=args.id,
        epoch_width=epoch_width,
        seed=seed,
        train_split=train_split,
        backtest_split=backtest_split,
        regression_lookback=regression_lookback,
        regression_predictions=regression_predictions,
        model_discovery_steps=model_discovery_steps,
        idle_minutes_on_position_close=idle_minutes_on_position_close,
    )




# Classification Training Data for Unit Tests
# Alters the current Epoch's configuration and sets the classification training data ID that will be used by
# the Unit Tests
elif args.action == "classification_training_data_ut":
    # Init the Epoch
    Epoch.init()

    # Print the progress
    print("1/1) Setting the Class. Training Data ID for Unit Tests in the Epoch's Configuration...")

    # Finally, update the file
    Epoch.set_classification_training_data_id_ut(args.training_data_file_name.replace(".json", ""))




# Classification Training Data
# Alters the current Epoch's configuration and sets the selected classification training data ID
elif args.action == "classification_training_data":
    # Init the Epoch
    Epoch.init()

    # Print the progress
    print("1/1) Setting the Class. Training Data ID in the Epoch's Configuration...")

    # Update the file
    Epoch.set_classification_training_data_id(args.training_data_file_name.replace(".json", ""))




# Export Epoch
# Compiles all the neccessary data and outputs a zip file which is then processed by Epoca's Infrastructure.
elif args.action == "export":
    print("\n\nEXPORT EPOCH RUNNING\n")





# Unknown Action
else:
    raise ValueError(f"The provided action could not be processed: {str(args.action)}")





# End of Script
Utils.endpoint_footer(endpoint_name)