from typing import Union, Dict, Any
from modules._types import IEpochConfig
from modules.utils.Utils import Utils


# package.json file
package_file: Dict[str, Any] = Utils.read("package.json")



# Class
class Configuration:
    """Configuration Class

    This singleton handles the management for configuration files.

    Class Properties:

    Instance Properties:
    """

    # Paths
    DIR_PATH: str = "config"
    EPOCH_PATH: str = f"{DIR_PATH}/epoch.json"

    # Version
    VERSION: str = package_file["version"]







    @staticmethod
    def get_epoch_config(allow_empty: bool = False) -> Union[IEpochConfig, None]:
        """Retrieves the Epoch Configuration.

        Args:
            allow_empty: bool
                If enabled, an error won't be raised in case the file doesn't exist
                and instead returns None.

        Returns:
            Union[IEpochConfig, None]
        """
        return Utils.read(Configuration.EPOCH_PATH, allow_empty=allow_empty)






    @staticmethod
    def update_epoch_config(new_config: IEpochConfig) -> None:
        """Updates the current Epoch Configuration.

        Args:
            new_config: IEpochConfig
                The new config to be set on the file
        """
        Utils.write(Configuration.EPOCH_PATH, data=new_config, indent=4)

    