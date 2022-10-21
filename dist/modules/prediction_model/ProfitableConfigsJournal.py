from typing import List, Union
from modules._types import IProfitableConfigurationsJournal, IPredictionModelMinifiedConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch





class ProfitableConfigsJournal:
    """ProfitableConfigsJournal Class

    This class handles the temporary storing of profitable prediction models configurations.
    In the event where the process has to be stopped for any reason, this class is constantly
    storing its progress whenever a profitable config is found. Once the process is resumed,
    it will start from point very close to the point in which the process was interrupted.

    Class Properties:
        ...

    Instance Properties:
        batch_file_name: str
            The file name of the batch that is going to be put through the process.
        path: str
            The path of the journal file.
        current_index: int
            The index in which the last profitable configuration was found.
        configs: List[IPredictionModelMinifiedConfig]
            The list of profitable configurations found for the given batch.
    """





    def __init__(self, batch_file_name: str):
        """Initializes the ProfitableConfigsJournal Instance.
        
        Args:
            batch_file_name: str
                The name of the configurations batch file.
        """
        # Init the path
        self.path: str = Epoch.PATH.profitable_configs_journal()

        # Init the batch name
        self.batch_file_name: str = batch_file_name

        # Retrieve the current state of the journal
        journal: Union[IProfitableConfigurationsJournal, None] = Utils.read(self.path, True)

        # Check if the journal is set and it has the same batch name
        if isinstance(journal, dict) and journal.get("batch_file_name") == self.batch_file_name:
            # Init the index
            self.current_index: int = journal["current_index"]

            # Init the configs
            self.configs: List[IPredictionModelMinifiedConfig] = journal["configs"]

        # Otherwise, set the default state
        else:
            # Init the index
            self.current_index: int = 0

            # Init the configs
            self.configs: List[IPredictionModelMinifiedConfig] = []










    def save_profitable_config(self, config_index: int, config: IPredictionModelMinifiedConfig) -> None:
        """When a profitable configuration is found, is added to the local properties
        and also stored in the file as well as the index in which the config was found.

        Args:
            config_index: int
                The index in which the profitable config was found.
            config: IPredictionModelMinifiedConfig
                The configuration that is considered to be profitable.
        """
        # Set the current index
        self.current_index = config_index

        # Add the config to the list
        self.configs.append(config)

        # Finally, update the journal
        Utils.write(self.path, {
            "batch_file_name": self.batch_file_name,
            "current_index": self.current_index,
            "configs": self.configs,
        })











    def clear_journal(self) -> None:
        """When a batch has been fully processed, this function is invoked
        in order to delete the journal file.
        """
        # Make sure there is a journal, otherwise skip the action
        if Utils.file_exists(self.path):
            Utils.remove_file(self.path)
