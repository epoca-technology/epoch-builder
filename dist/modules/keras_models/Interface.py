from typing import List, Any



# Keras Model Interface
# Regression and Classification implement the following interface
# in order to ensure compatibility across any of the processes.
class KerasModelInterface:
    # Performs a prediction based provided features
    def predict(self, *args,**kwargs) -> List[float]:
        raise NotImplementedError("KerasModel.predict has not been implemented.")

    # Retrieves the configuration of the Keras Model
    def get_config(self) -> Any:
        raise NotImplementedError("KerasModel.get_config has not been implemented.")






