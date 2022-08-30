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
	 * @param trainable_model_types: object
	 */
	constructor(cluster_server, cluster_path, trainable_model_types) { 		
		// Initialize the servers instance
		this.cluster_server = cluster_server;

		// Initialize the epoch path instance
		this.cluster_path = cluster_path;

		// Initialize the trainable model types
		this.trainable_model_types = trainable_model_types;
	}







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






	/**
	 * Displays the list of trainable model types based on the provided
	 * configuration and returns the selected one.
	 * @param mode: string "all"|"regression"|"classification"
	 * @returns Promise<string>
	 */
	async trainable_model_type(mode) {
		// Present the list
		console.log(" ");
		const answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a type of model", choices: this.trainable_model_types[mode]}
		]);

		// Finally, return the answer
		return answer["value"];
	}





	/**
	 * Displays all the neccessary forms in order to pack the hyperparams 
	 * command data.
	 * @returns Promise<object>
	 */
	async hyperparams() {
		// Retrieve the model type
		const model_type = await this.trainable_model_type("all");

		// Init the classification training data (Only used in classifications)
		var training_data_file_name = "";

		// Init the default batch size to be displayed
		var default_batch_size = "";

		// If it is a classification, retrieve the file name and set the default batch size
		if (this.trainable_model_types["classification"].includes(model_type)) {
			// Set the default batch size to 60 (Classification default batch size)
			default_batch_size = "60";

			// Retrieve the classification training data file
			training_data_file_name = await this.classification_training_data_file_name();
		} 
		
		// Otherwise, set the batch size to 30 (Regression default batch size)
		else { default_batch_size = "30" }
	
		// Present the batch size form
		console.log(" ");
		const batch_size = await inquirer.prompt([
			{
				type: "input", name: "value", message: `Enter the batch size (Optional: defaults to ${default_batch_size})`, 
				validate(value) {
					if (typeof value != "string" || isNaN(value) || Number(value) < 5 || Number(value) > 200) {
						return "Please enter a valid batch size. It can be an int ranging 5 - 200.";
					} else { return true }
				}
			}
		]);

		// Finally, return the packed values
		return { model_type: model_type, training_data_file_name: training_data_file_name, batch_size: batch_size["value"] }
	}





	/**
	 * Based on the provided type of model, it will put together the training config
	 * category and configuration file.
	 * @param trainable_model_type 
	 * @returns object { category: string, config_file_name: string }
	 */
	async training_config(trainable_model_type) {
		// Retrieve the list of categories
		const categories = FileSystem.get_path_content(this.cluster_path.training_configs(true, trainable_model_type));

		// Present the list of categories
		console.log(" ");
		const category_answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a category", choices: categories.directories}
		]);

		// Retrieve the list of configuration files
		let config_files = FileSystem.get_path_content(
			this.cluster_path.training_configs(true, trainable_model_type, category_answer["value"])
		);
		config_files.files.sort((a, b) => { return this.get_batch_number(a) > this.get_batch_number(b) ? 1: -1});

		// Present the list of config files
		console.log(" ");
		const config_file_answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a configuration file", choices: config_files.files}
		]);

		// Return the packed results
		return { category: category_answer["value"], config_file_name: config_file_answer["value"] }
	}



	/**
	 * Retrieves the batch number from a given config file name.
	 * @param config_file_name: string
	 * @returns number
	 */
	get_batch_number(config_file_name) { try { return Number(config_file_name.split("_").at(-2)) } catch (e) { return 0 } }








	/**
	 * Displays the text input for the selected model ids.
	 * @returns Promise<string>
	 */
	async selected_model_ids() {
		// Present the input
		console.log(" ");
		const answer = await inquirer.prompt([{
			type: "input", name: "value", message: "Enter the selected model ids separated by commas", 
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
		}]);

		// Finally, return the answer
		return answer["value"];
	}




	/**
	 * Displays the form for the classification training data creation.
	 * @returns Promise<object>
	 */
	async classification_training_data() {
		// Retrieve the regression selections
		const { directories, files} = FileSystem.get_path_content(this.cluster_path.regression_selection(true));

		// Present the form
		console.log(" ");
		const form = await inquirer.prompt([
			{type: "list", name: "regression_selection_file_name", message: "Select the regression selection", choices: files},
			{
				type: "input", name: "description", message: "Enter the description", 
				validate(value) {
					if (typeof value != "string" || value.length < 5 || value.length > 100000) {
						return "Please enter a valid description.";
					} else { return true }
				}
			},
			{
				type: "input", name: "steps", message: "Enter the steps", 
				validate(value) {
					if (typeof value != "string" || isNaN(value) || Number(value) < 0 || Number(value) > 10) {
						return "Please enter valid steps. It can be an int ranging 0 - 10.";
					} else { return true }
				}
			},
			{type: "list", name: "include_rsi", message: "Include RSI?", choices: ["No", "Yes"]},
			{type: "list", name: "include_aroon", message: "Include AROON?", choices: ["No", "Yes"]}
			
		]);

		// Finally, return the values
		return {
			regression_selection_file_name: form["regression_selection_file_name"],
			description: form["description"],
			steps: form["steps"],
			include_rsi: form["include_rsi"],
			include_aroon: form["include_aroon"]
		}
	}




	/**
	 * Displays the list of classification training data files.
	 * @returns Promise<string>
	 */
	 async classification_training_data_file_name() {
		// Retrieve the files
		const { directories, files} = FileSystem.get_path_content(this.cluster_path.classification_training_data(true));

		// Present the list
		console.log(" ");
		const answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a classification training data", choices: files}
		]);

		// Finally, return the answer
		return answer["value"];
	}
	




	/**
	 * Displays the list of backtest configurations.
	 * @returns Promise<string>
	 */
	async backtest_config() {
		// Retrieve the backtest configurations
		const { directories, files} = FileSystem.get_path_content(this.cluster_path.backtests(true, "configurations"));

		// Present the list
		console.log(" ");
		const answer = await inquirer.prompt([
			{type: "list", name: "value", message: "Select a backtest configuration", choices: files}
		]);

		// Finally, return the answer
		return answer["value"];
	}






	/**
	 * Displays the epoch management form.
	 * @returns Promise<object>
	 */
	 async epoch_management() {
		// Init the args
		let args = {
			action: "",
			id: "",
			epoch_width: "",
			seed: "",
			train_split: "",
			backtest_split: "",
			regression_lookback: "",
			regression_predictions: "",
			model_discovery_steps: "",
			idle_minutes_on_position_close: "",
			training_data_file_name: "",
			model_ids: ""
		};

		// Present the list of actions
		console.log(" ");
		const action = await inquirer.prompt([
			{type: "list", name: "value", message: "Select an action", choices: [
				"create", "classification_training_data_ut", "classification_training_data", "export"
			]}
		]);
		args.action = action["value"];

		// Handle the creation of an epoch
		if (args.action == "create") {
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
					type: "input", name: "epoch_width", message: "Enter the epoch's width (Optional: defaults to 24)", 
					validate(value) {
						if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 6 || Number(value) > 48)) {
							return "Please enter a valid epoch width. It can be an int ranging 6 - 48.";
						} else { return true }
					}
				},
				{
					type: "input", name: "seed", message: "Enter the random seed (Optional: defaults to 60184)", 
					validate(value) {
						if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 1 || Number(value) > 100000000)) {
							return "Please enter a valid epoch width. It can be an int ranging 6 - 48.";
						} else { return true }
					}
				},
				{
					type: "input", name: "train_split", message: "Enter the train split (Optional: defaults to 0.85)", 
					validate(value) {
						if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 0.6 || Number(value) > 0.95)) {
							return "Please enter a valid train split. It can be a float ranging 0.6 - 0.95.";
						} else { return true }
					}
				},
				{
					type: "input", name: "backtest_split", message: "Enter the backtest split (Optional: defaults to 0.3)", 
					validate(value) {
						if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 0.2 || Number(value) > 0.6)) {
							return "Please enter a valid backtest split. It can be a float ranging 0.2 - 0.6.";
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
					type: "input", name: "model_discovery_steps", message: "Enter the model discovery steps (Optional: defaults to 5)", 
					validate(value) {
						if (typeof value == "string" && value.length && (isNaN(value) || Number(value) < 1 || Number(value) > 20)) {
							return "Please enter a valid regression predictions. It can be an int ranging 1 - 20.";
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

			// Set the args
			args.id = form["id"];
			args.epoch_width = form["epoch_width"];
			args.seed = form["seed"];
			args.train_split = form["train_split"];
			args.backtest_split = form["backtest_split"];
			args.regression_lookback = form["regression_lookback"];
			args.regression_predictions = form["regression_predictions"];
			args.model_discovery_steps = form["model_discovery_steps"];
			args.idle_minutes_on_position_close = form["idle_minutes_on_position_close"];
		}

		// Handle the setting of the classification training data
		else if (args.action == "classification_training_data_ut" || args.action == "classification_training_data") {
			args.training_data_file_name = await this.classification_training_data_file_name();
		}

		// Handle the exporting of the epoch
		else if (args.action == "export") {
			args.model_ids = await this.selected_model_ids();
		}

		// Finally, return the args
		return args;
	}







	/**
	 * Displays the database management form.
	 * @param server 
	 * @returns Promise<object>
	 */
	async db_management(server) {
		// Init the ip
		let ip = "";

		// Present the list of actions
		console.log(" ");
		const action = await inquirer.prompt([
			{type: "list", name: "value", message: "Select an action", choices: ["summary", "backup", "restore", "update_host_ip"]}
		]);

		// Check if the update_host_ip action was selected
		if (action["value"] == "update_host_ip") {
			// Populate the default ip
			var default_ip = "";
			// Check if the server is the master or is not a part of the cluster
			if (server.is_master || !server.is_cluster) { default_ip = server.ip }

			// If the server is in the cluster but is not the master, find the master's ip
			else { default_ip = this.cluster_server.get_server_master().ip }

			// Present the ip input
			console.log(" ");
			const ip_input = await inquirer.prompt([{
				type: "input", name: "value", message: "Enter the Master's IP",
				default() { return default_ip },
				validate(value) {
					if (typeof value != "string" || value.length < 9) {
						return "Please enter a valid IP.";
					} else { return true }
				}
			}]);

			// Update the final ip
			ip = ip_input["value"];
		}

		// Finally, return the packed values
		return { action: action["value"], ip: ip }
	}
}






// Export the modules
export { ClusterInput };