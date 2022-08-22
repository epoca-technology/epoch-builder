from typing import Dict
from inquirer import List as InquirerList, prompt
from subprocess import Popen
from modules._types import IDatabaseSummary
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.database.Database import Database, DB_CONNECTION_CONFIG


# Initialize the Epoch
Epoch.init()


# Welcome
print("DATABASE MANAGEMENT\n")


# Process Menu
process: Dict[str, str] = prompt([
    InquirerList("id", message="Select an action to execute", choices=[
        "View Database Summary",
        "Backup Database",
        "Restore Database"
    ])
])


# Database Summary
# Extracts a summary of the tables and the data stored so far.
if process["id"] == "View Database Summary":
    # Retrieve the summary
    s: IDatabaseSummary = Database.get_summary()

    # Create the function that will print the sizes
    def format_size(size: float) -> str:
        return str(round(size/1073741824, 4)) + " Gbs"

    # Output it
    print("\nDATABASE SUMMARY\n")
    print(s["version"])
    print(f"Name: {s['connection_config']['database']}")
    print(f"User: {s['connection_config']['user']}")
    print(f"Host IP: {s['connection_config']['host_ip']}")
    print(f"Port: {s['connection_config']['port']}")
    print(f"Password: {s['connection_config']['password']}")
    print(f"Total Size: {format_size(s['size'])}")
    print(f"\nTables ({len(s['tables'])}):")
    for t in s["tables"]:
        print(f"{t['name']}: {format_size(t['size'])}")
    print(f"\nTest Tables ({len(s['test_tables'])}):")
    for tt in s["test_tables"]:
        print(f"{tt['name']}: {format_size(tt['size'])}")

    print("\n\nDATABASE SUMMARY COMPLETED")





# Database Backup
# Generates a compressed backup file and places it in the backup directory.
elif process["id"] == "Backup Database":
    print("\nDATABASE BACKUP RUNNING")
    # Init the backup directory
    backup_path: str = Database.DB_MANAGEMENT_PATH + "/backup"
    backup_name: str = str(Utils.get_time()) + ".dump"

    # Create the backup directory if it does not exist
    Utils.make_directory(backup_path)

    # Init the command
    command: str = f"pg_dump \
        -U {DB_CONNECTION_CONFIG['user']} \
            -h {DB_CONNECTION_CONFIG['host_ip']} \
                -p {DB_CONNECTION_CONFIG['port']} \
                    -d {DB_CONNECTION_CONFIG['database']} \
                        -f {backup_path}/{backup_name} -Fc"

    # Add the required env vars and execute the command
    proc = Popen(command, shell=True, env={
        'PGPASSWORD': DB_CONNECTION_CONFIG['password']
    })
    proc.wait()
    print("\n\nDATABASE BACKUP COMPLETED")





# Database Restore
# Reads the restore directory and restores the compressed backup file located in it.
else:
    print("\nDATABASE RESTORE RUNNING")
    # Init the restore path
    restore_path: str = Database.DB_MANAGEMENT_PATH + "/restore"

    # Make sure the restore path exists, otherwise create it and provide information
    if not Utils.directory_exists(restore_path):
        Utils.make_directory(restore_path)
        raise ValueError(f"The backup dump to be restored must be placed in {restore_path}")

    # Extract the names of the databases in the path. Make sure there is exactly 1 dump
    _, backup_names = Utils.get_directory_content(restore_path, only_file_ext=".dump")
    if len(backup_names) != 1:
        raise ValueError(f"There must be 1 backup dump file to be restored in {restore_path}")

    # Clean all the tables and reinitialize them fresh
    Database.delete_tables()
    Database.initialize_tables()

    # Init the command
    command: str = f"pg_restore --clean \
        -U {DB_CONNECTION_CONFIG['user']} \
            -h {DB_CONNECTION_CONFIG['host_ip']} \
                -p {DB_CONNECTION_CONFIG['port']} \
                    -d {DB_CONNECTION_CONFIG['database']} {restore_path}/{backup_names[0]}"

    # Add the required env vars and execute the command
    proc = Popen(command, shell=True, env={
        'PGPASSWORD': DB_CONNECTION_CONFIG['password']
    })
    proc.wait()
    print("\n\nDATABASE RESTORE COMPLETED")