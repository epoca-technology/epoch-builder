





/**
 * Cluster Path
 * This class handles the retrieval of any path within the project.
 * IMPORTANT: If this class is initialized with an undefined epoch_id,
 * invoking _epoch_path will trigger an error.
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
	requirements(local) { return this._path(local, "requirements.txt") }




	/**
	 * Retrieves the path for the package.json file
	 * @param local: boolean
	 * @returns string
	 */
	package_json(local) { return this._path(local, "package.json") }





	/**
	 * Retrieves the path for the nohup.out file
	 * @param local: boolean
	 * @returns string
	 */
	nohup_logs(local) { return this._path(local, "nohup.out") }









	/* Root Paths */



	/**
	 * Retrieves the path for the config directory.
	 * @param local: boolean
	 * @returns string
	 */
	config(local) { return this._path(local, "config") }



	 
	/**
	 * Retrieves the path for the database management directory.
	 * @param local: boolean
	 * @returns string
	 */
	db_management(local) { return this._path(local, "db_management") }




	 
	/**
	 * Retrieves the path for the candlesticks management directory.
	 * @param local: boolean
	 * @returns string
	 */
	candlesticks(local) { return this._path(local, "candlesticks") }





	/**
	 * Retrieves the path for the dist directory.
	 * @param local: boolean
	 * @returns string
	 */
	dist(local) { return this._path(local, "dist") }





	/**
	 * Retrieves the path for the unit tests directory.
	 * @param local: boolean
	 * @returns string
	 */
	unit_tests(local) { return this._path(local, "dist/tests") }








	/* Epoch Paths */



	/**
	 * Retrieves the path for the backtest configurations, results or the 
	 * root directory.
	 * @param local: boolean
	 * @param dir_name?: string "configurations"|"results"
	 * @returns string
	 */
	backtests(local, dir_name = undefined) { 
		// Check if the dir name was provided
		if (typeof dir_name == "string") { return this._epoch_path(local, `backtests/${dir_name}`) }

		// Otherwise, return the root directory
		else { return this._epoch_path(local, "backtests")  }
	}





	/**
	 * Retrieves the path for the batched training certificates. If no
	 * type is provided, it will return the base directory.
	 * @param local: boolean
	 * @param trainable_model_type?: string
	 * @returns string
	 */
	batched_training_certificates(local, trainable_model_type = undefined) {
		// Check if the model type was provided
		if (typeof trainable_model_type == "string") { 
			return this._epoch_path(local, `batched_training_certificates/${trainable_model_type}`); 
		}

		// Otherwise, return the root directory
		else { return this._epoch_path(local, "batched_training_certificates") }
	}




	/**
	 * Retrieves the path for the active models directory.
	 * @param local: boolean
	 * @returns string
	 */
	models(local) { return this._epoch_path(local, "models") }





	/**
	 * Retrieves the path for the models bank. If no type is provided, it will 
	 * return the base directory.
	 * @param local: boolean
	 * @param trainable_model_type?: string
	 * @returns string
	 */
	models_bank(local, trainable_model_type = undefined) {
		// Check if the model type was provided
		if (typeof trainable_model_type == "string") { 
			return this._epoch_path(local, `models_bank/${trainable_model_type}`) 
		}

		// Otherwise, return the root directory
		else { return this._epoch_path(local, "models_bank") }
	}




	/**
	 * Retrieves the path for the regression selection directory.
	 * @param local: boolean
	 * @returns string
	 */
	regression_selection(local) { return this._epoch_path(local, "regression_selection") }





	/**
	 * Retrieves the path for the classification training data directory.
	 * @param local: boolean
	 * @returns string
	 */
	classification_training_data(local) { return this._epoch_path(local, "classification_training_data") }
	




	/**
	 * Retrieves the path for the training configurations directory. If the model
	 * type is not provided, it will return the root of the directory.
	 * @param local: boolean
	 * @param trainable_model_type?: string
	 * @param category?: string
	 * @returns string
	 */
	training_configs(local, trainable_model_type = undefined, category = undefined) { 
		// Check if the model type was provided
		if (typeof trainable_model_type == "string") {
			// Check if the category was provided
			if (typeof category == "string") {
				return this._epoch_path(local, `training_configs/${trainable_model_type}/${category}`);
			}

			// Otherwise, return the model specific path
			else {
				return this._epoch_path(local, `training_configs/${trainable_model_type}`);
			}
		} 
		
		// Otherwise, return the root directory
		else {
			return this._epoch_path(local, "training_configs") 
		}
	}







	

	/* Path Builders */






	/**
	 * If a path is provided, it will prepend the epoch's id to it. Otherwise, 
	 * returns the root directory of the epoch.
	 * @param local: boolean
	 * @param path?: string
	 */
	_epoch_path(local, path = undefined) {
		// Make sure the Epoch ID was initialized
		if (typeof this.epoch_id != "string") {
			throw new Error("The Epoch's ID was not set in the ClusterPath because the configuration file could not be loaded.");
		}

		// If the path was provided, prepend the epoch's id to it
		if (typeof path == "string") { return this._path(local, `${this.epoch_id}/${path}`)} 

		// Otherwise, return the root directory of the epoch
		else { return this._path(local, this.epoch_id) }
	}






	/**
	 * Builds the final local or external path that will be used for any action.
	 * If no path is provided, it returns the root path of the epoch builder.
	 * @param local: boolean
	 * @param path?: string
	 * @returns string
	 */
	_path(local, path = undefined) { 
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