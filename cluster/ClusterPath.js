import { FileSystem } from "./FileSystem.js";





/**
 * Cluster Path
 * This class handles the retrieval of any Cluster Path or contents.
 * IMPORTANT: If this class is initialized with an undefined epoch_id, most 
 * of its methods will throw an error.
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





	/**
	 * Retrieves the path for the batched training certificates. If no
	 * type is provided, it will return the base directory.
	 * @param local: boolean
	 * @param trainable_model_type?: string
	 * @returns string
	 */
	batched_training_certificates(local, trainable_model_type) {
		if (typeof this.epoch_id != "string") throw new Error(this.EPOCH_ID_NOT_SET_ERROR);

		// Check if the model type was provided
		if (typeof trainable_model_type == "string") { 
			return this._path(local, `${this.epoch_id}/batched_training_certificates/${trainable_model_type}`); 
		}

		// Otherwise, return the root directory
		else { return this._path(local, `${this.epoch_id}/batched_training_certificates`) }
	}



	/**
	 * Retrieves the path for the models bank. If no type is provided, it will 
	 * return the base directory.
	 * @param local: boolean
	 * @param trainable_model_type?: string
	 * @returns string
	 */
	 models_bank(local, trainable_model_type) {
		if (typeof this.epoch_id != "string") throw new Error(this.EPOCH_ID_NOT_SET_ERROR);

		// Check if the model type was provided
		if (typeof trainable_model_type == "string") { 
			return this._path(local, `${this.epoch_id}/models_bank/${trainable_model_type}`) 
		}

		// Otherwise, return the root directory
		else { return this._path(local, `${this.epoch_id}/models_bank`) }
	}





	/**
	 * Retrieves the path for the database management directory.
	 * @param local: boolean
	 * @returns string
	 */
	 database_management(local) { return this._path(local, "db_management") }








	/**
	 * Builds the final local or external path that will be used for any action.
	 * If no path is provided, it returns the root path of the epoch builder.
	 * @param local: boolean
	 * @param path?: string
	 * @returns string
	 */
	_path(local, path) { 
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