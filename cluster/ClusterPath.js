





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
	// Epoch ID Not Set Error
	EPOCH_ID_NOT_SET_ERROR = "The Epoch's ID was not set in the ClusterPath because the configuration file could not be loaded.";


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












	/* Epoch Paths */





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
	 * Retrieves the path for the training configurations directory.
	 * @param local: boolean
	 * @param trainable_model_type: string
	 * @returns string
	 */
	training_configs(local, trainable_model_type) { return this._epoch_path(local, `${trainable_model_type}_training_configs`) }





	/**
	 * Retrieves the path for the backtest configurations directory.
	 * @param local: boolean
	 * @returns string
	 */
	backtest_configurations(local) { return this._epoch_path(local, "backtest_configurations") }






	/**
	 * Retrieves the path for the backtest results directory.
	 * @param local: boolean
	 * @returns string
	 */
	backtest_results(local) { return this._epoch_path(local, "backtest_results") }









	

	/* Path Builders */






	/**
	 * If a path is provided, it will prepend the epoch's id to it. Otherwise, 
	 * returns the root directory of the epoch.
	 * @param local: boolean
	 * @param path?: string
	 */
	_epoch_path(local, path = undefined) {
		// Make sure the Epoch ID was initialized
		if (typeof this.epoch_id != "string") throw new Error(this.EPOCH_ID_NOT_SET_ERROR);

		// If the path was provided, prepend the epoch's id to it
		if (typeof path == "string") { return this._path(local, `${this.epoch_id}/${path}`)} 

		// Otherwise, return the root directory of the epoch
		else { return this._path(this.epoch_id) }
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