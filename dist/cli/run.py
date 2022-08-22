from typing import List, Dict, Union
from subprocess import Popen
from inquirer import List as InquirerList, Text, prompt
from tensorflow import config
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration

# Welcome
gpus_available: int = len(config.list_physical_devices("GPU"))
Utils.clear_terminal()
if gpus_available > 0:
    print(f"EPOCH BUILDER\n")
else:
    print(f"EPOCH BUILDER GPU\n")


# Database Host IP
# Checks if the Database Host IP has been set. If not, it will prompt a form
# in order to set it prior to running the process.
db_host_ip: Union[str, None] = Configuration.get_db_host_ip()
if db_host_ip is None:
    db_host_ip: Dict[str, str] = prompt([Text("value", f"Enter the Database Host IP")])
    Configuration.set_db_host_ip(db_host_ip["value"])



# Processes
print(" ")
processes: Dict[str, str] = {
    "Backtest": "run_backtest.py",
    "Classification Training Data": "run_classification_training_data.py",
    "Classification Training": "run_classification_training.py",
    "Database Management": "run_db_management.py",
    "Epoch Management": "run_epoch_management.py",
    "Hyperparams": "run_hyperparams.py",
    "Merge Training Certificates": "run_merge_training_certificates.py",
    "Regression Selection": "run_regression_selection.py",
    "Regression Training": "run_regression_training.py",
    "Update Database Host IP": "run_update_db_host_ip.py",
    "Unit Tests": ""
}
processes_answer: List[Dict[str, str]] = prompt([
    InquirerList("process", message="Select the process to run", choices=processes.keys())
])


# Init the process
Utils.clear_terminal()
proc: Popen

# Execute a CLI Script
if processes_answer["process"] != "Unit Tests":
    proc = Popen(f"python3 dist/cli/{processes[processes_answer['process']]}", shell=True)

# Otherwise, execute the Unit Tests
else:
    proc = Popen("python3 -m unittest discover -s dist/tests -p '*_test.py'", shell=True)

# Wait for the process to complete
proc.wait()