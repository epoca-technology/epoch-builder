





/**
 * Cluster Path
 * This class handles the retrieval of any path within the project.
 * IMPORTANT: If this class is initialized with an undefined epoch_id,
 * invoking epoch_path will trigger an error.
 * 
 * Instance Properties
 * 	local_path: string
 * 	epoch_id: string|undefined
 */
 class ClusterPath {



	/**
	 * Initializes a Cluster Path Instance
	 * @param local_path: string
	 * @param epoch_id: string|undefined
	 */
	constructor(local_path, epoch_id) { 
		// Init the local path
		this.local_path = local_path;

		// Init the Epoch's ID
		this.epoch_id = epoch_id;
	}




	/* Root File Paths */



	/**
	 * Retrieves the path for the requirements.txt file
	 * @param local: boolean
	 * @returns string
	 */
	requirements(local) { return this.path(local, "requirements.txt") }




	/**
	 * Retrieves the path for the package.json file
	 * @param local: boolean
	 * @returns string
	 */
	package_json(local) { return this.path(local, "package.json") }





	/**
	 * Retrieves the path for the nohup.out file
	 * @param local: boolean
	 * @returns string
	 */
	nohup_logs(local) { return this.path(local, "nohup.out") }









	/* Root Paths */



	/**
	 * Retrieves the path for the config directory.
	 * @param local: boolean
	 * @returns string
	 */
	config(local) { return this.path(local, "config") }




	 
	/**
	 * Retrieves the path for the candlesticks management directory.
	 * @param local: boolean
	 * @returns string
	 */
	candlesticks(local) { return this.path(local, "candlesticks") }





	/**
	 * Retrieves the path for the dist directory.
	 * @param local: boolean
	 * @returns string
	 */
	dist(local) { return this.path(local, "dist") }





	/**
	 * Retrieves the path for the unit tests directory.
	 * @param local: boolean
	 * @returns string
	 */
	unit_tests(local) { return this.path(local, "dist/tests") }









	/* Epoch Paths */





	/**
	 * Retrieves the path for the  regression training configurations directory. If no
	 * category is provided, it will return the root directory.
	 * @param local: boolean
	 * @param category?: string
	 * @returns string
	 */
	regression_training_configs(local, category = undefined) { 
		// Check if the category was provided
		if (typeof category == "string") {
			return this.epoch_path(local, `regression_training_configs/${category}`);
		}

		// Otherwise, return the root path
		else {
			return this.epoch_path(local, "regression_training_configs");
		}
	}





	/**
	 * Retrieves the path for the regression batched training certificates.
	 * @param local: boolean
	 * @returns string
	 */
	regression_batched_certificates(local) { 
		return this.epoch_path(local, "regression_batched_certificates") 
	}






	/**
	 * Retrieves the path for the regressions directory. If no regression id is
	 * provided, it will return the root path.
	 * @param local: boolean
	 * @param id?: string
	 * @returns string
	 */
	regressions(local, id = undefined) { 
		// Check if the id was provided
		if (typeof id == "string") {
			return this.epoch_path(local, `regressions/${id}`);
		}

		// Otherwise, return the root path
		else {
			return this.epoch_path(local, "regressions");
		}
	}





	/**
	 * Retrieves the path for the root prediction models directory.
	 * @param local: boolean
	 * @returns string
	 */
	prediction_models(local) { 
		return this.epoch_path(local, "prediction_models") 
	}




	/**
	 * Retrieves the path for the prediction models configurations
	 * directory.
	 * @param local: boolean
	 * @returns 
	 */
	prediction_models_configs(local) { 
		return this.epoch_path(local, "prediction_models/configs") 
	}






	/**
	 * Retrieves the path for the profitable prediction models configurations
	 * directory.
	 * @param local: boolean
	 * @returns 
	 */
	prediction_models_profitable_configs(local) { 
		return this.epoch_path(local, "prediction_models/profitable_configs") 
	}






	

	/* Path Builders */






	/**
	 * If a path is provided, it will prepend the epoch's id to it. Otherwise, 
	 * returns the root directory of the epoch.
	 * @param local: boolean
	 * @param path?: string
	 */
	epoch_path(local, path = undefined) {
		// Make sure the Epoch ID was initialized
		if (typeof this.epoch_id != "string") {
			throw new Error("The Epoch's ID was not set in the ClusterPath because the configuration file could not be loaded.");
		}

		// If the path was provided, prepend the epoch's id to it
		if (typeof path == "string") { return this.path(local, `${this.epoch_id}/${path}`)} 

		// Otherwise, return the root directory of the epoch
		else { return this.path(local, this.epoch_id) }
	}






	/**
	 * Builds the final local or external path that will be used for any action.
	 * If no path is provided, it returns the root path of the epoch builder.
	 * @param local: boolean
	 * @param path?: string
	 * @returns string
	 */
	path(local, path = undefined) { 
		// Check if a path was provided
		if (typeof path == "string") {
			return local ? `${this.local_path}/epoch-builder/${path}`: `epoch-builder/${path}`
		}
		
		// Otherwise, return the root directory
		else {
			return local ? `${this.local_path}/epoch-builder`: `epoch-builder`
		}
	 }
}













// Export the modules
export { ClusterPath };