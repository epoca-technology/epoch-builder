from typing import List, Union, Any, Tuple
from os import makedirs, listdir, system, name as os_name
from os.path import exists, isfile, dirname, splitext
from shutil import rmtree, move, copy
from json import load, dumps
from time import time
from datetime import datetime
from uuid import UUID, uuid4
from modules._types import IFileExtension




class Utils:
    """Utils Class

    This singleton provides a series of functionalities that simplify development and 
    provide consistency among modules.

    Number Helpers:
        alter_number_by_percentage(value: float, percent: float) -> float
        get_percentage_change(old_value: float, new_value: float) -> float
        get_percentage_out_of_total(value: float, total: float) -> float

    Time Helpers:
        get_time() -> int
        from_milliseconds_to_seconds(milliseconds: Union[int, float]) -> int
        from_seconds_to_milliseconds(seconds: Union[int, float]) -> int
        from_milliseconds_to_minutes(ms: Union[int, float]) -> int
        from_date_string_to_milliseconds(date_str: str) -> int
        from_milliseconds_to_date_string(ms: int) -> str
        add_minutes(timestamp_ms: Union[int, float], minutes: int) -> int

    UUID Helpers:
        generate_uuid4() -> str
        is_uuid4(uuid: str) -> bool

    File System Helpers:
        directory_exists(path: str) -> bool
        file_exists(path: str) -> bool
        move_file_or_dir(source: str, destination: str) -> None
        copy_file_or_dir(source: str, destination: str) -> None
        make_directory(path: str) -> None
        remove_directory(path: str) -> None
        get_directory_content(path: str, only_file_ext: Union[IFileExtension, None] = None) -> Tuple[List[str], List[str]]
        read(path: str, allow_empty: bool = False) -> Any
        write(path: str, data: Any, timestamp_file_name: bool = False, indent: Union[int, None] = None) -> None

    Misc Helpers:
        prettify_model_id(id: str) -> str
        clear_terminal() -> None
    """









    ####################
    ## Number Helpers ##
    ####################






    @staticmethod
    def alter_number_by_percentage(value: float, percent: float) -> float:
        """Alters a number based on given percentage. For example, if value is 100 and percent
        is 50 it will return 150. On the other hand, if value is 100 and percent is -50 it will
        return 50.

        Args:
            value: float
                The number that will be altered.
            percent: float 
                The percentage that will be applied to the value.

        Returns:
            float
        """
        # Init the new value
        new_value = value

        # Handle an increase
        if percent > 0:
            new_value = ((percent / 100) + 1) * value

        # Handle a decrease
        elif percent < 0:
            new_value = -(((percent * -1) / 100) - 1) * value
        
        # Return the altered number
        return round(new_value, 2)







    @staticmethod
    def get_percentage_change(old_value: float, new_value: float) -> float:
        """Calculates the percentage change a value has experienced.

        Args:
            old_value: float
                The original number to calculate the % change for.
            new_value: float
                The new state of the original number.

        Returns:
            float
        """
        # If the old value is zero, the percentage change cannot be calculated
        if old_value == 0:
            return 0
            
        # Init the change
        change: float = 0.0

        # Handle an increase
        if new_value > old_value:
            increase: float = new_value - old_value
            change = (increase / old_value) * 100

        # Handle a decrease
        elif old_value > new_value:
            decrease: float = old_value - new_value
            change = -((decrease / old_value) * 100)

        # Return the change
        return round(change if change >=-100 else -100, 2)









    @staticmethod
    def get_percentage_out_of_total(value: float, total: float) -> float:
        """Calculates the percentage representation of a value based on the total.
        For example, if value is 20 and total is 200 the result is 10%

        Args:
            value: float
                The value to calculate the % representation for.
            total: float
                The total number of existing elements

        Returns:
            float
        """
        return round((value * 100) / total, 2)
















    ##################
    ## Time Helpers ##
    ##################







    @staticmethod
    def get_time() -> int:
        """Retrieves the current time in milliseconds. Equivalent of Javascript's Date.now().

        Args:
            None

        Returns:
            int
        """
        return Utils.from_seconds_to_milliseconds(time())








    @staticmethod
    def from_milliseconds_to_seconds(milliseconds: Union[int, float]) -> int:
        """Converts a milli seconds value into seconds. Notice that it will round
        decimals downwards in case of a float.

        Args:
            milliseconds: Union[int, float]
                The timestamp in milliseconds to be converted to seconds.

        Returns:
            int
        """
        return int(milliseconds / 1000)






    @staticmethod
    def from_seconds_to_milliseconds(seconds: Union[int, float]) -> int:
        """Converts a seconds value into milliseconds. Notice that it will round 
        decimals downwards in case of a float.

        Args:
            seconds: Union[int, float]
                The seconds timestamp to be converted to milliseconds.

        Returns:
            int
        """
        return int(seconds * 1000)





    @staticmethod
    def from_milliseconds_to_minutes(ms: Union[int, float]) -> int:
        """Converts milliseconds into minutes. Notice that it will round 
        decimals downwards in case of a float.

        Args:
            ms: Union[int, float]
                The milliseconds to be converted to minutes.

        Returns:
            int
        """
        return round(ms / 60000)






    @staticmethod
    def from_date_string_to_milliseconds(date_str: str) -> int:
        """Converts a date string into a milliseconds timestamp format.

        Args:
            date_str: str
                The format must be as follows: DD/MM/YYYY. For example: '30/02/2020'

        Returns:
            int
        """
        # Split the date arguments
        date_split: List[str] = date_str.split('/')

        # Initialize the DateTime Instance
        dt: datetime = datetime(int(date_split[2]), int(date_split[1]), int(date_split[0]))

        # Return the timestamp in milliseconds
        return Utils.from_seconds_to_milliseconds(dt.timestamp())







    @staticmethod
    def from_milliseconds_to_date_string(ms: int) -> str:
        """Converts a timestamp in milliseconds into a readeable string.

        Args:
            ms: int
                Timestamp in milliseconds.

        Returns:
            str
        """
        return datetime.fromtimestamp(Utils.from_milliseconds_to_seconds(ms)).strftime("%d/%m/%Y, %H:%M:%S")






    @staticmethod
    def add_minutes(timestamp_ms: Union[int, float], minutes: int) -> int:
        """Adds minutes to a given timestamp. The output of this function is a 
        timestamp in milliseconds with the added minutes.

        Args:
            timestamp_ms: Union[int, float]
                The original timestamp that will be incremented.
            minutes: int 
                The number of minutes that will be added to the timestamp.

        Returns:
            int
        """
        return int(timestamp_ms + (Utils.from_seconds_to_milliseconds(60) * minutes))
    















    ##################
    ## UUID Helpers ##
    ##################




    @staticmethod
    def generate_uuid4() -> str:
        """Generates a random Universally Unique Identifier.

        Returns:
            str
        """
        return str(uuid4())





    @staticmethod
    def is_uuid4(uuid: str) -> bool:
        """Verifies if a provided uuid is valid.

        Returns:
            bool
        """
        uuid_obj: Union[UUID, None] = None
        try:
            uuid_obj = UUID(str(uuid), version=4)
        except ValueError:
            return False
        return str(uuid_obj) == uuid














    #########################
    ## File System Helpers ##
    #########################




    
    @staticmethod
    def directory_exists(path: str) -> bool:
        """Checks if a given directory path exists.

        Args:
            path: str
                The path to be checked for existance.

        Returns:
            bool
        """
        return exists(path)





    @staticmethod
    def file_exists(path: str) -> bool:
        """Checks if a given file path exists.

        Args:
            path: str
                The path to be checked for existance.

        Returns:
            bool
        """
        return isfile(path)









    @staticmethod
    def move_file_or_dir(source: str, destination: str) -> None:
        """Moves a directory or file from source to destination

        Args:
            source: str
                The path that will be moved to the destination.
            destination: str
                The path in which the source will be moved to.
        """
        # Firstly make sure the source exists
        if not Utils.file_exists(source) and not Utils.directory_exists(source):
            raise RuntimeError(f"The file/dir cannot be moved because the source does not exist: {source}")

        # Finally, move the file/dir
        move(source, destination)







    @staticmethod
    def copy_file_or_dir(source: str, destination: str) -> None:
        """Copies a directory or file from source to destination

        Args:
            source: str
                The path that will be copied
            destination: str
                The path where it will be pasted
        """
        # Firstly make sure the source exists
        if not Utils.file_exists(source) and not Utils.directory_exists(source):
            raise RuntimeError(f"The file/dir cannot be copied because the source does not exist: {source}")

        # Finally, move the file/dir
        copy(source, destination)






    @staticmethod
    def make_directory(path: str) -> None:
        """Creates a directory at a given path if it doesnt already exist.

        Args:
            path: str
                The path in which the directory should be created.
        """
        if not Utils.directory_exists(path):
            makedirs(path)





    @staticmethod
    def remove_directory(path: str) -> None:
        """Removes a directory and its contents.

        Args:
            path: str
                The path of the directory that will be removed.

        Raises:
            RuntimeError:
                If the directory does not exist
        """
        # Make sure the directory exists
        if not Utils.directory_exists(path):
            raise RuntimeError(f"The directory {path} cannot be removed because it does not exist.")

        # Remove the directory
        rmtree(path)







    @staticmethod
    def get_directory_content(path: str, only_file_ext: Union[IFileExtension, None] = None) -> Tuple[List[str], List[str]]:
        """Retrieves all the directories and files located in the
        provided path.

        Args:
            path: str
                The path of the directory
            only_file_ext: Union[IFileExtension, None]
                If an extension is provided, it will filter all files with not 
                matching format.

        Returns:
            Tuple[List[str], List[str]]
            (directories, files)
        
        Raises:
            RuntimeError:
                If the directory does not exist.
        """
        # Init values
        directories: List[str] = []
        files: List[str] = []

        # Make sure the directory exists
        if not Utils.directory_exists(path):
            raise RuntimeError(f"The contents of the directory {path} cannot be retrieved because it does not exist.")

        # Iterate over each item in the directory
        for item in listdir(path):
            # Check if it is a file
            if isfile(f"{path}/{item}"):
                files.append(item)
            
            # Otherwise, it is a directory
            else:
                directories.append(item)
        
        # Filter the files if applies
        if isinstance(only_file_ext, str):
            files = list(filter(lambda x: only_file_ext in x, files))

        # Finally, return the contents
        return sorted(directories), sorted(files)






    @staticmethod
    def read(path: str, allow_empty: bool = False) -> Any:
        """Reads a file located at a given path and returns
        its contents.

        Args:
            path: str
                The path in which the file is located.
            allow_empty: bool
                If enabled, the function won't raise an error if the file
                does not exist.
        
        Returns:
            Any

        Raises:
            RuntimeError:
                If the file does not exist and allow_empty is set to False.
        """
        # Check if the file exists
        if Utils.file_exists(path):
            # Split the path into path name and extension
            _, extension = splitext(path)

            # Read the file according to its format
            if extension == ".json":
                return load(open(path))
            else:
                return open(path).read()

        # Otherwise, check if an error needs to raised
        else:
            if allow_empty:
                return None
            else:
                raise RuntimeError(f"The file {path} does not exist.")






    @staticmethod
    def write(path: str, data: Any, timestamp_file_name: bool = False, indent: Union[int, None] = None) -> None:
        """Writes a file on given path.

        Args:
            path: str
                The path of the file that will be written. Note that if the file
                exists, it will overwrite it.
            data: Any
                The data to be stored in the file. If it is a JSON file, the data must
                be compatible.
            timestamp_file_name: bool
                If enabled, it will append the current timestamp to the file name in the
                following way: my_file.json -> my_file_1657887972213.json
            indent: Union[int, None]
                The indenting to be applied on the JSON File. Defaults to no indenting.
        """
        # Make sure the directory exists
        dir_name: str = dirname(path)
        if not Utils.directory_exists(dir_name):
            Utils.make_directory(dir_name)

        # Split the path into path name and extension
        path_name, extension = splitext(path)

        # Check if the file name needs to be timestamped
        if timestamp_file_name:
            path_name += f"_{Utils.get_time()}"

        # Write the File based on its format
        with open(path_name + extension, "w") as file_wrapper:
            if extension == ".json":
                file_wrapper.write(dumps(data, indent=indent))
            else:
                file_wrapper.write(data)


















    ##################
    ## Misc Helpers ##
    ##################





    @staticmethod
    def prettify_model_id(id: str) -> str:
        """Given a model id, it will prettify it so it can be viewed
        in the console properly.

        Returns:
            str
        """
        return id if len(id) < 23 else f"{id[0:20]}..."







    @staticmethod
    def clear_terminal() -> None:
        """Clears the system's terminal.
        """
        system("cls" if os_name == "nt" else "clear")