import inquirer from "inquirer";




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
			{ type: "list", name: "value", message: "Select the type of process", loop: false, choices: Object.keys(process_menu) }
		]);

		// Retrieve the list of processes within the category
		console.log(" ");
		const process = await inquirer.prompt([
			{ type: "list", name: "value", message: "Select a process", loop: false, choices: process_menu[category["value"]] }
		]);

		// Finally, return the process id
		return process["value"];
	}



	


	/**
	 * Retrieves the list of servers and displays the list
	 * according to the provided configuration. Once a server
	 * is selected, the server object is returned.
	 * @param include_localhost: boolean
	 * @param include_all: boolean
	 * @param online_only: boolean
	 * @param available_only: boolean
	 * @returns Promise<object>
	 */
	async server(include_localhost = false, include_all = false, online_only = false, available_only = false) {
		// Retrieve the list of servers
		const servers = await this.cluster_server.list_servers(include_localhost, include_all);

		// Iterate over each server and populate the list of choices
		let choices = [];
		for (let server of servers) {
			// Populate the state of the choice
			let disabled_state = undefined;
			if (online_only && !server.is_online) { disabled_state = "Offline" }
			else if (available_only && !server.is_available) { disabled_state = "Busy" }

			// Push the choice to the list
			choices.push({ name: server.name, disabled: disabled_state })
		}

		// Present the list
		console.log(" ");
		const server_answer = await inquirer.prompt([{type: "list", name: "value", message: "Select a server", loop: false, choices: choices}]);

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
			{type: "list", name: "value", message: "Select a type of model", loop: false, choices: this.trainable_model_types[mode]}]);

		// Finally, return the answer
		return answer["value"];
	}


}





// Export the modules
export { ClusterInput };