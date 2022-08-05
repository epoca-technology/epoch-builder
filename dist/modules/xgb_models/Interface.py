from typing import List, Any



# XGB Model Interface
# Regression and Classification implement the following interface
# in order to ensure compatibility across any of the processes.
class XGBModelInterface:
    # Performs a prediction based provided features
    def predict(self, *args,**kwargs) -> List[float]:
        raise NotImplementedError("XGBModel.predict has not been implemented.")

    # Retrieves the configuration of the XGB Model
    def get_config(self) -> Any:
        raise NotImplementedError("XGBModel.get_config has not been implemented.")






