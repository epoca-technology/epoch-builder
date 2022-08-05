from pandas import DataFrame, concat
from modules.types import ICompressedTrainingData, IClassificationDatasets
from modules.classification_training_data.TrainingDataCompression import decompress_training_data




def make_datasets(training_data: ICompressedTrainingData, train_split: float) -> IClassificationDatasets:
    """Splits the Classification Training Data into train and test dataframes.

    Args:
        training_data: ICompressedTrainingData
            The Training Data to be decompressed and split.

    Returns:
        IClassificationDatasets
        (train_x, train_y, test_x, test_y)
    """
    # Decompress the training data
    df: DataFrame = decompress_training_data(training_data)
    
    # Initialize the total rows and the split size
    rows: int = df.shape[0]
    split: int = int(rows * train_split)

    # Initialize the features dfs
    train_x: DataFrame = df[:split]
    test_x: DataFrame = df[split:]

    # Initialize the labels dfs
    train_y: DataFrame = concat([train_x.pop(x) for x in ["up", "down"]], axis=1)
    test_y: DataFrame = concat([test_x.pop(x) for x in ["up", "down"]], axis=1)

    # Return the packed dfs
    return train_x, train_y, test_x, test_y