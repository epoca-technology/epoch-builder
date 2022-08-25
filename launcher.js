import { spawn } from "child_process";
import inquirer from "inquirer";




/**
 * 
 */
async function main() {
	// Welcome
	console.clear();
	console.log("EPOCH BUILDER LAUNCHER\n\n");

	// Ask for the process to execute
	const process = await ask_for_process();
	console.log(process);
	await connect_to_a_server();
}





/**
 * Constants
 */



/**
 * Process Categories
 * Due to the high number of processes, in order to simplify the usage of the 
 * launcher, they are grouped in 4 categories.
 */
 const PROCESS_CATEGORIES = ["Server", "Epoch Builder", "Push", "Pull"];

 // Server Processes
 const SERVER_PROCESSES = [
	 "Connect to a server",
	 "View server logs",
	 "Reboot server",
	 "Shutdown server",
	 "Shutdown all servers",
	 "Install SSH Keys on a server"
 ];
 
 // Epoch Builder Processes
 const EPOCH_BUILDER_PROCESSES = [
	 "Backtest",
	 "Classification Training Data",
	 "Classification Training",
	 "Database Management",
	 "Epoch Management",
	 "Hyperparams",
	 "Merge Training Certificates",
	 "Regression Selection",
	 "Regression Training",
	 "Update Database Host IP",
	 "Unit Tests",
 ];
 
 // Push Processes
 const PUSH_PROCESSES = [
	 "Push Configuration",
	 "Push Database Management",
	 "Push Candlesticks",
	 "Push Active Models",
	 "Push Regression Selection",
	 "Push Classification Training Data",
	 "Push Training Configurations",
	 "Push Backtest Configurations",
	 "Push Backtest Results",
	 "Push Epoch",
	 "Push Epoch Builder"
 ];
 
 // Pull Processes
 const PULL_PROCESSES = [
	 "Pull Database Management",
	 "Pull Regression Selection",
	 "Pull Classification Training Data",
	 "Pull Backtest Results",
	 "Pull Trained Models"
 ];






function connect_to_a_server() {
	return execute("ssh", ["-i", "~/.ssh/eb_id_rsa", "epoca-worker-01@192.168.1.236"]);
}









/**
 * Executes a given command and subscribes to its events. The promise is 
 * resolved once the process indicates it and all the accumulated data is
 * returned (if any). In case no data is accumulated, undefined will be returned.
 * @param command: string
 * @param args: string[]
 * @param options: SpawnOptionsWithoutStdio
 * @returns Promise<string|undefined>
 */
function execute(command, args, options = { stdio: "inherit" }) {
	return new Promise((resolve, reject) => {
		// Start the process
		const ls = spawn(command, args, options);

		// Init the data
		let data = "";

		// Subscribe to the stdout data event if possible
		if (ls.stdout) ls.stdout.on("data", stdout_data => { data += stdout_data});
		
		// Subscribe to the stdeer data event if possible
		if (ls.stderr) ls.stdout.on("data", stderr_data => { data += stderr_data});
		
		// Subscribe to the error event
		ls.on("error", (error) => { reject(error) });
		
		// Subscribe to the close event
		ls.on("close", code => {
			// Make sure the process exited with code 0
			if (code == 0) {
				resolve(data.length > 0 ? data: undefined)
			} 
			
			// Otherwise, handle the error
			else {
				reject(`The ${command} process exited with the error code ${code}`)
			}
		});
	})
}











/**
 * Prompts
 */





/**
 * Asks the user for a process to execute.
 * @returns string
 */
async function ask_for_process() {
	// Retrieve the category of the process
	const category = await inquirer.prompt([
		{ type: "list", name: "value", message: "Select a category", loop: false, choices: PROCESS_CATEGORIES }
	]);

	// Retrieve the list of processes within the category
	console.log(" ");
	const process = await inquirer.prompt([
		{ type: "list", name: "value", message: "Select a category", loop: false, choices: get_category_processes(category["value"]) }
	]);

	// Finally, return the process name
	return process["value"];
}



/**
 * Given a category name, it retrieves the list of processes within it.
 * @param category_name: string
 * @returns string[]
 */
function get_category_processes(category_name) {
	switch(category_name) {
		case "Server":
			return SERVER_PROCESSES;
		case "Epoch Builder":
			return EPOCH_BUILDER_PROCESSES;
		case "Push":
			return PUSH_PROCESSES;
		case "Pull":
			return PULL_PROCESSES;
		default:
			throw new Error(`The category name ${category_name} does not have processes in it.`)
	}
}











/**
 * Process Execution
 * If successful, the process will terminate with a status of 0. Otherwise, 
 * prints the error and terminates with a status of 1.
 */
main().then(() => { process.exit(0) }).catch(e => { console.error(e); process.exit(1); })