import inquirer from "inquirer";
import { FileSystem } from "./FileSystem.js";



 /**
  * Cluster Input
  * This class handles all the user input that is required in the Cluster.
  * 
  * Instance Properties
  * 	cluster_server: ClusterServer
  * 	cluster_path: ClusterPath
  */
class ClusterInput {


	/**
	 * Initializes a User Input Instance
	 * @param cluster_server: ClusterServer
	 * @param cluster_path: ClusterPath
	 */
	constructor(cluster_server, cluster_path) { 		
		// Initialize the servers instance
		this.cluster_server = cluster_server;

		// Initialize the epoch path instance
		this.cluster_path = cluster_path;
	}






	/* Shared Inputs */



	/**
	 * Displays the process menu and takes the user through the selection of 
	 * a process id to run.
	 * @param process_menu: object
	 * @returns Promise<string>
	 */
	 async process_id(process_menu) {
		// Retrieve the category of the process
		const category = await inquirer.prompt([
			{ type: "list", name: "value", message: "Select the type of process", choices: Object.keys(process_menu) }
		]);

		// Retrieve the list of processes within the category
		console.log(" ");
		const process = await inquirer.prompt([
			{ type: "list", name: "value", message: "Select a process", choices: process_menu[category["value"]] }
		]);

		// Finally, return the process id
		return process["value"];
	}



	


	/**
	 * Retrieves the list of servers and displays the list
	 * according to the provided configuration. Once a server
	 * is selected, the server object is returned.
	 * @param include_localhost?: boolean
	 * @param include_all?: boolean
	 * @param online_only?: boolean
	 * @param available_only?: boolean
	 * @param busy_only?: boolean
	 * @returns Promise<object>
	 */
	async server(include_localhost = false, include_all = false, online_only = false, available_only = false, busy_only = false) {
		// Retrieve the list of servers
		const servers = await this.cluster_server.list_servers(include_localhost, include_all);

		// Iterate over each server and populate the list of choices
		let choices = [];
		for (let server of servers) {
			// Populate the state of the choice
			let disabled_state = undefined;
			if (online_only && !server.is_online) { disabled_state = "Offline" }
			else if (available_only && !server.is_available) { disabled_state = "Busy" }
			else if (busy_only && server.is_available) { disabled_state = "Not Busy" }

			// Push the choice to the list
			choices.push({ name: server.name, disabled: disabled_state })
		}

		// Present the list
		console.log(" ");
		const server_answer = await inquirer.prompt([{type: "list", name: "value", message: "Select a server", choices: choices}]);

		// Finally, return the server
		return this.cluster_server.get_server(server_answer["value"]);
	}




	


	/* Epoch Builder Specific */




	/**
	 * Displays the epoch creation form.
	 * @returns Promise<object>
	 */
	 async create_epoch() {
		// Present the form
		const form = await inquirer.prompt([
			{
				type: "input", name: "id", message: "Enter the ID", 
				validate(value) {
					if (typeof value != "string" || value.length < 4 || value[0] != "_") {
						return "Please enter a valid epoch id.";
					} else { return true }
				}
			},
			{
				type: "input", name: "seed", message: "Enter the random seed (Optional: defaults to 60184)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 1 || Number(value) > 100000000)) {
						return "Please enter a valid epoch width. It can be an int ranging 1 - 100000000.";
					} else { return true }
				}
			},
			{
				type: "input", name: "epoch_width", message: "Enter the epoch's width (Optional: defaults to 24)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 6 || Number(value) > 48)) {
						return "Please enter a valid epoch width. It can be an int ranging 6 - 48.";
					} else { return true }
				}
			},
			{
				type: "input", name: "sma_window_size", message: "Enter the SMA Window Size (Optional: defaults to 100)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 10 || Number(value) > 300)) {
						return "Please enter a valid sma window size. It can be an int ranging 10 - 300.";
					} else { return true }
				}
			},
			{
				type: "input", name: "train_split", message: "Enter the train split (Optional: defaults to 0.75)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 0.6 || Number(value) > 0.95)) {
						return "Please enter a valid train split. It can be a float ranging 0.6 - 0.95.";
					} else { return true }
				}
			},
			{
				type: "input", name: "validation_split", message: "Enter the validation split (Optional: defaults to 0.2)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 0.15 || Number(value) > 0.4)) {
						return "Please enter a valid validation split. It can be a float ranging 0.15 - 0.4.";
					} else { return true }
				}
			},
			{
				type: "input", name: "regression_lookback", message: "Enter the regression lookback (Optional: defaults to 128)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 32 || Number(value) > 512)) {
						return "Please enter a valid regression lookback. It can be an int ranging 32 - 512.";
					} else { return true }
				}
			},
			{
				type: "input", name: "regression_predictions", message: "Enter the regression predictions (Optional: defaults to 32)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 32 || Number(value) > 256)) {
						return "Please enter a valid regression predictions. It can be an int ranging 32 - 256.";
					} else { return true }
				}
			},
			{
				type: "input", name: "exchange_fee", message: "Enter the exchange fee (Optional: defaults to 0.065)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 0.02 || Number(value) > 0.2)) {
						return "Please enter a valid position size. It can be a float ranging 0.02-0.2.";
					} else { return true }
				}
			},
			{
				type: "input", name: "position_size", message: "Enter the position size (Optional: defaults to 10000)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 100 || Number(value) > 100000000)) {
						return "Please enter a valid position size. It can be a float ranging 100 - 100000000.";
					} else { return true }
				}
			},
			{
				type: "input", name: "leverage", message: "Enter the leverage (Optional: defaults to 5)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 1 || Number(value) > 10)) {
						return "Please enter a valid leverage. It can be an int ranging 1 - 10.";
					} else { return true }
				}
			},
			{
				type: "input", name: "idle_minutes_on_position_close", message: "Enter the idle minutes on position_close (Optional: defaults to 30)", 
				validate(value) {
					if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 0 || Number(value) > 1000)) {
						return "Please enter a valid idle minutes on position_close. It can be an int ranging 0 - 1000.";
					} else { return true }
				}
			}
		]);

		// Finally, return the args
		return {
			seed: form["seed"],
			id: form["id"],
			epoch_width: form["epoch_width"],
			sma_window_size: form["sma_window_size"],
			train_split: form["train_split"],
			validation_split: form["validation_split"],
			regression_lookback: form["regression_lookback"],
			regression_predictions: form["regression_predictions"],
			exchange_fee: form["exchange_fee"],
			position_size: form["position_size"],
			leverage: form["leverage"],
			idle_minutes_on_position_close: "idle_minutes_on_position_close"
		};
	}








	/**
	 * Displays the forms that collect the category and the batch file.
	 * @returns Promise<object> { category: string, batch_file_name: string }
	 */
	async regression_training_configs() {
		// Retrieve the list of categories
		const categories = FileSystem.get_path_content(this.cluster_path.regression_training_configs(true));

		// Present the list of categories
		const category_answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a category", choices: categories.directories}
		]);

		// Retrieve the list of configuration files
		let config_files = FileSystem.get_path_content(
			this.cluster_path.regression_training_configs(true, category_answer["value"])
		);
		config_files.files.sort((a, b) => { return this.get_batch_number(a) > this.get_batch_number(b) ? 1: -1});

		// Present the list of config files
		console.log(" ");
		const config_file_answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a configuration file", choices: config_files.files}
		]);

		// Return the packed results
		return { category: category_answer["value"], batch_file_name: config_file_answer["value"] }
	}


	





	/**
	 * Displays the form that collects the information in order to build
	 * the prediction models.
	 * @returns Promise<string>
	 */
	async initialize_prediction_models() {
		// Present the input
		const args = await inquirer.prompt(
			[
				{
					type: "input", name: "regression_ids", message: "Enter the regression ids separated by commas", 
					validate(value) {
						// Make sure a value has been set
						if (typeof value != "string" || !value.length) {
							return "Please enter a valid list of model ids.";
						}
		
						// Otherwise, make sure that at least 1 model has been provided
						else {
							const model_ids = value.split(",");
							if (model_ids.length >= 20) {
								return true;
							} else {
								return "A minimum of 20 regression ids must be provided.";
							}
						}
					}
				}
			]
		);

		// Finally, return the answer
		return args["regression_ids"]
	}









	/**
	 * Displays the forms that collects the batch_file_name where
	 * profitable model configurations will be extracted.
	 * @returns Promise<string>
	 */
	async find_profitable_configs() {
		// Retrieve the list of configuration files
		let config_files = FileSystem.get_path_content(this.cluster_path.prediction_models_configs(true));
		config_files.files.sort((a, b) => { return this.get_batch_number(a) > this.get_batch_number(b) ? 1: -1});

		// Present the list of config files
		console.log(" ");
		const config_file_answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a configuration file", choices: config_files.files}
		]);

		// Return the packed results
		return config_file_answer["value"]
	}






	/**
	 * Displays the epoch creation form.
	 * @returns Promise<string>
	 */
	async build_prediction_models() {
		// Present the form
		const form = await inquirer.prompt([
			{
				type: "input", name: "limit", message: "Enter the limit of prediction models that will be placed in the build", 
				validate(value) {
					if (isNaN(value) || Number(value) < 10 || Number(value) > 1000) {
						return "Please enter a valid limit. It can be an int ranging 1 - 1000.";
					} else { return true }
				}
			},
		]);

		// Finally, return the limit
		return form["limit"]
	}









	/**
	 * Displays the form that collects the information in order to export an 
	 * Epoch
	 * @returns Promise<string>
	 */
	 async export_epoch() {
		// Present the input
		const args = await inquirer.prompt(
			[
				{
					type: "input", name: "ids", message: "Enter the prediction model ids separated by commas", 
					validate(value) {
						// Make sure a value has been set
						if (typeof value != "string" || !value.length) {
							return "Please enter a valid list of model ids.";
						}
		
						// Otherwise, make sure that at least 1 model has been provided
						else {
							const model_ids = value.split(",");
							if (model_ids.length) {
								return true;
							} else {
								return "A minimum of 1 model id must be provided.";
							}
						}
					}
				}
			]
		);

		// Finally, return the answer
		return args["ids"]
	}










	/* Misc Helpers */






	/**
	 * Retrieves the batch number from a given config file name.
	 * @param config_file_name: string
	 * @returns number
	 */
	 get_batch_number(config_file_name) { try { return Number(config_file_name.split("_").at(-2)) } catch (e) { return 0 } }
}






// Export the modules
export { ClusterInput };