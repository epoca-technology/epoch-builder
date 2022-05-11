from typing import Union, List, Any
from keras import Sequential
from modules.keras_models import IKerasModelSummary, IKerasModelLayer





def get_summary(model: Union[Sequential, Any]) -> IKerasModelSummary:
    """Based on a given trained model, it will extract all the relevant information and return
    the full summary.

    Args:
        model: ...
            The trained model which the summary will be built based on.
        include_weights: bool
            If True it will add the weights attribute to the summary. Keep in mind that this 
            attribute may be large.
    Returns:
        IKerasModelSummary
    """
    # Init Values
    layers: List[IKerasModelLayer] = []
    trainable_params: int = 0
    non_trainable_params: int = 0

    # Build the layers' data
    for layer in model.layers:
        # Calculate the layer's params
        layer_params: int = layer.count_params()

        # Append the layer to the list
        layers.append({
            "name": layer.name,
            "params": layer_params,
            "input_shape": layer.input_shape,
            "output_shape": layer.output_shape,
            "trainable": layer.trainable,
        })

        # Increment the param counters accordingly
        if layer.trainable:
            trainable_params = trainable_params + layer_params
        else:
            non_trainable_params = non_trainable_params + layer_params
    
    # Finally, return the summary
    return {
        "model_class": model.__class__.__name__,
        "optimizer_config": {key: str(value) for key, value in model.optimizer.get_config().items()},
        "loss_config": {key: str(value) for key, value in model.loss.get_config().items()},
        "metrics": model.metrics_names,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layers": layers,
        "total_params": model.count_params(),
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }