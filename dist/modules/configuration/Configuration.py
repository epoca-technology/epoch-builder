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
    DB_HOST_IP_PATH: str = f"{DIR_PATH}/db_host_ip.txt"
    EPOCH_PATH: str = f"{DIR_PATH}/epoch.json"

    # Version
    VERSION: str = package_file["version"]




    ######################
    ## Database Host IP ##
    ######################




    @staticmethod
    def get_db_host_ip() -> Union[str, None]:
        """Retrieves the IP of the Database Host if exists. Otherwise, returns None.

        Returns:
            Union[str, None]
        """
        return Utils.read(Configuration.DB_HOST_IP_PATH, allow_empty=True)






    @staticmethod
    def set_db_host_ip(ip: str) -> None:
        """Sets an IP in the DB Host configuration file.

        Args:
            ip: str
                The IP to be set as the Database Host.
        
        Raises:
            ValueError: If the provided ip is invalid.
        """
        # Make sure the provided IP is valid
        if not isinstance(ip, str) or len(ip) < 9:
            ValueError("The provided DB Host IP is invalid.")

        # Finally, create/update the file
        Utils.write(Configuration.DB_HOST_IP_PATH, ip)










    #########################
    ## Epoch Configuration ##
    #########################






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

    