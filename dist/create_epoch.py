from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch



# CREATE EPOCH
# Args:
#   --seed? "60184"
#   --id "_ALPHA"
#   --epoch_width? "24"
#   --sma_window_size? "100"
#   --train_split? "0.75"
#   --validation_split? "0.2"
#   --regression_lookback? "128"
#   --regression_predictions? "32"
#   --position_size? "10000"
#   --leverage? "3"
#   --idle_minutes_on_position_close? "30"
endpoint_name: str = "CREATE EPOCH"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--seed", dest="seed", nargs='?')
parser.add_argument("--id", dest="id")
parser.add_argument("--epoch_width", dest="epoch_width", nargs='?')
parser.add_argument("--sma_window_size", dest="sma_window_size", nargs='?')
parser.add_argument("--train_split", dest="train_split", nargs='?')
parser.add_argument("--validation_split", dest="validation_split", nargs='?')
parser.add_argument("--regression_lookback", dest="regression_lookback", nargs='?')
parser.add_argument("--regression_predictions", dest="regression_predictions", nargs='?')
parser.add_argument("--position_size", dest="position_size", nargs='?')
parser.add_argument("--leverage", dest="leverage", nargs='?')
parser.add_argument("--idle_minutes_on_position_close", dest="idle_minutes_on_position_close", nargs='?')
args = parser.parse_args()


# Init the values or fill them with defaults
seed: int = int(args.seed) if args.seed.isdigit() else Epoch.DEFAULTS["seed"]
epoch_width: int = int(args.epoch_width) if args.epoch_width.isdigit() else Epoch.DEFAULTS["epoch_width"]
sma_window_size: int = int(args.sma_window_size) if args.sma_window_size.isdigit() else Epoch.DEFAULTS["sma_window_size"]
train_split: float = float(args.train_split) if args.train_split.isdigit() else Epoch.DEFAULTS["train_split"]
validation_split: float = float(args.validation_split) if args.validation_split.isdigit() else Epoch.DEFAULTS["validation_split"]
regression_lookback: int = int(args.regression_lookback) if args.regression_lookback.isdigit() else Epoch.DEFAULTS["regression_lookback"]
regression_predictions: int = int(args.regression_predictions) if args.regression_predictions.isdigit() else Epoch.DEFAULTS["regression_predictions"]
position_size: float = float(args.position_size) if args.position_size.isdigit() else Epoch.DEFAULTS["position_size"]
leverage: int = int(args.leverage) if args.leverage.isdigit() else Epoch.DEFAULTS["leverage"]
idle_minutes_on_position_close: int = int(args.idle_minutes_on_position_close) if args.idle_minutes_on_position_close.isdigit() else Epoch.DEFAULTS["idle_minutes_on_position_close"]


# Finally, create the epoch
Epoch.create(
    seed=seed,
    id=args.id,
    epoch_width=epoch_width,
    sma_window_size=sma_window_size,
    train_split=train_split,
    validation_split=validation_split,
    regression_lookback=regression_lookback,
    regression_predictions=regression_predictions,
    position_size=position_size,
    leverage=leverage,
    idle_minutes_on_position_close=idle_minutes_on_position_close
)




# End of Script
Utils.endpoint_footer(endpoint_name)