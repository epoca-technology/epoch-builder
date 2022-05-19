from pandas import DataFrame
from modules.classification import ICompressedTrainingData







def compress_training_data(df: DataFrame) -> ICompressedTrainingData:
    """Breaks down a DataFrame into a list of columns and rows.

    Args:
        df: DataFrame
            The df to be compressed.

    Returns:
        ICompressedTrainingData
    """
    return {"columns": df.columns.values.tolist(), "rows": df.values.tolist()}









def decompress_training_data(data: ICompressedTrainingData) -> DataFrame:
    """Given a compressed training data dict, it will convert it into a DataFrame.

    Args:
        data: ICompressedTrainingData
            The data to be decompressed.

    Returns:
        DataFrame
    """
    return DataFrame(data=data["rows"], columns=data["columns"])