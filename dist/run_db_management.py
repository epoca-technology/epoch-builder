from typing import List, Dict
from os import makedirs, listdir
from os.path import exists, isfile, join
from inquirer import List as InquirerList, prompt
from subprocess import Popen
from modules.database import Database, DB_CONNECTION_CONFIG, IDatabaseSummary
from modules.utils import Utils


## Database Summary ##


def _display_db_summary() -> None:
    """Displays the summary of the Database.
    """
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





## Database Backup ##

def _backup_db() -> None:
    """Backs up the entire Database into the db management directory.
    """
    print("\nDATABASE BACKUP RUNNING")
    # Init the backup directory
    backup_path: str = Database.DB_MANAGEMENT_PATH + "/backup"
    backup_name: str = str(Utils.get_time()) + ".dump"

    # Create the backup directory if it does not exist
    if not exists(backup_path):
        makedirs(backup_path)

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
    print("\nDATABASE BACKUP COMPLETED")





## Database Restore ##


def _restore_db() -> None:
    print("\nDATABASE RESTORE RUNNING")
    # Init the restore path
    restore_path: str = Database.DB_MANAGEMENT_PATH + "/restore"

    # Make sure the restore path exists, otherwise create it and provide information
    if not exists(restore_path):
        makedirs(restore_path)
        raise ValueError(f"The dump to be restored must be placed in {restore_path}")

    # Extract the names of the databases in the path. Make sure there is exactly 1 dump
    names: List[str] = list(filter(lambda f: ".dump" in f, [f for f in listdir(restore_path) if isfile(join(restore_path, f))]))
    if len(names) != 1:
        raise ValueError(f"There must be 1 dump file to be restored in {restore_path}")

    # Clean all the tables and reinitialize them fresh
    Database.delete_tables()
    Database.initialize_tables()

    # Init the command
    command: str = f"pg_restore --clean \
        -U {DB_CONNECTION_CONFIG['user']} \
            -h {DB_CONNECTION_CONFIG['host_ip']} \
                -p {DB_CONNECTION_CONFIG['port']} \
                    -d {DB_CONNECTION_CONFIG['database']} {restore_path}/{names[0]}"

    # Add the required env vars and execute the command
    proc = Popen(command, shell=True, env={
        'PGPASSWORD': DB_CONNECTION_CONFIG['password']
    })
    proc.wait()
    print("\nDATABASE RESTORE COMPLETED")







## CLI ##
print("DATABASE MANAGEMENT")
SUMMARY: str = "View Database Summary"
BACKUP: str = "Backup Database"
RESTORE: str = "Restore Database"
questions = [InquirerList("action", message="Select an action to execute", choices=[SUMMARY, BACKUP, RESTORE])]
answer: Dict[str, str] = prompt(questions)
if answer['action'] == SUMMARY:
    _display_db_summary()
elif answer['action'] == BACKUP:
    _backup_db()
elif answer['action'] == RESTORE:
    _restore_db()
else:
    raise ValueError(f"The provided action could not be processed: {str(answer)}")