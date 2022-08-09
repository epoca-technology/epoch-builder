from modules._types import IDiscovery




def normalize(change: float, discovery: IDiscovery) -> float:
    """Given a predicted price change and a regression's discovery, it 
    will normalize the value to a range of -1|1. If the value is smaller
    than the minimum set in the discovery, it will return 0.

    Args:
        change: float
            The percentage change between the current price and the last
            predicted price.
        discovery: IDiscovery
            The regression's discovery results.

    Returns:
        float
    """
    # Retrieve the adjusted change
    adjusted_change: float = _calculate_adjusted_change(change, discovery)

    # Scale the increase change
    if adjusted_change > 0:
        return _scale(adjusted_change, discovery["increase_min"], discovery["increase_max"])
    
    # Scale the decrease change, keep in mind that the decrease data is in negative numbers.
    elif adjusted_change < 0:
        return -(_scale(-(adjusted_change), -(discovery["decrease_max"]), -(discovery["decrease_min"])))
    
    # Otherwise, return 0 as a sign of neutrality
    else:
        return 0





def _calculate_adjusted_change(change: float, discovery: IDiscovery) -> float:
    """Adjusts the provided change to the min and max values in the
    regression discovery.

    Args:
        change: float
        discovery: IDiscovery

    Returns:
        float
    """
    if change >= discovery["increase_min"] and change <= discovery["increase_max"]:
        return change
    elif change > discovery["increase_max"]:
        return discovery["increase_max"]
    elif change >= discovery["decrease_min"] and change <= discovery["decrease_max"]:
        return change
    elif change < discovery["decrease_min"]:
        return discovery["decrease_min"]
    else:
        return 0





def _scale(value: float, min: float, max: float) -> float:
    """Scales a prediction change based on the regression discovery's
    min and max values.

    Args:
        value: float
            The predicted price change that needs to be scaled.
        min: float
        max: float
            The minimum and maximum percentage changes extracted 
            from the regression discovery.
    """
    return round((value - min) / (max - min), 6)