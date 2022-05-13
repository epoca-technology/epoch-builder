from typing import List, Tuple, Dict
from pandas import DataFrame
from numpy import arange, ndarray, array, float32
from tensorflow import stack, Tensor, data as tfdata
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from modules.regression import ITrainingWindowGeneratorConfig





class TrainingWindowGenerator:
    """TrainingWindowGenerator Class

    This class manages the data windowing for the train, validation and test data.

    Class Properties:
        ...

    Instance Properties:
        train_df: DataFrame
            The Train DataFrame
        val_df: DataFrame
            The Validation DataFrame
        test_df: DataFrame
            The Test DataFrame
        label_columns: List[str]
            The name of the label's column
        label_columns_indices: Dict[str, int]
            A dictionary holding the label column names as well as the indices ({'c': 0})
        column_indices: Dict[str, int]
            A dictionary holding the feature and label column names as well as the indices ({'o': 0, 'h': 1, 'l': 2, 'c': 3})
        input_width: int
            The number of sequences that will be used to predict future values.
        label_width: int
            The number of predictions that will be generated.
        total_window_size: int
            The total size of the window (input_width + label_width)
        input_slice: slice
            The slicing instance that will be applied to the input_indices.
        input_indices: ndarray
            An array holding the input indices in the window. If a lookback of 50 is provided, the input_indices will be
            [0, 1 ... 48, 49]
        label_start: int
            The start point of the labels (total_window_size - label_width)
        labels_slice: slice
            The slicing instance that will be applied to the label_indices.
        label_indices: ndarray
            An array holding the label indices in the window. If a lookback of 50 and predictions of 5 is provided, the 
            label_indices will be [50, 51, 52, 53, 54]
        batch_size: int
            The size of the batch that will be used to build the train datasets.
        shuffle_data: bool
            If True, it will shuffle the train, val and test datasets prior to training.
    """


    def __init__(self, config: ITrainingWindowGeneratorConfig):
        """Initializes the Window Instance that will manage the data.

        Args:
            config: ITrainingWindowGeneratorConfig
                The configuration that will be used to initialize the window.
        """
        # Initialize the DataFrames
        self.train_df: DataFrame = config['train_df']
        self.val_df: DataFrame = config['val_df']
        self.test_df: DataFrame = config['test_df']

        # Work out the label column indices.
        self.label_columns: List[str] = config['label_columns']
        self.label_columns_indices: Dict[str, int] = { name: i for i, name in enumerate(self.label_columns) }
        self.column_indices: Dict[str, int] = { name: i for i, name in enumerate(self.train_df.columns) }

        # Work out the window parameters.
        self.input_width: int = config['input_width']
        self.label_width: int = config['label_width']

        # Calculate the total window size
        self.total_window_size: int = self.input_width + self.label_width

        # Initialize the Input Data
        self.input_slice: slice = slice(0, self.input_width)
        self.input_indices: ndarray = arange(self.total_window_size)[self.input_slice]

        # Initialize the Label Data
        self.label_start: int = self.total_window_size - self.label_width
        self.labels_slice: slice = slice(self.label_start, None)
        self.label_indices: ndarray = arange(self.total_window_size)[self.labels_slice]

        # Initialize the batch size
        self.batch_size: int = config['batch_size']

        # Initialize the data shuffling
        self.shuffle_data: bool = config['shuffle_data']





    ## Getters ##

    @property
    def train(self) -> tfdata.Dataset:
        return self.make_dataset(self.train_df)

    @property
    def val(self) -> tfdata.Dataset:
        return self.make_dataset(self.val_df)

    @property
    def test(self) -> tfdata.Dataset:
        return self.make_dataset(self.test_df)









    def split_window(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Given a list of consecutive inputs, the split_window method will convert them to 
        a window of inputs and a window of labels.

        Args:
            features: Tensor
                The input that will be windowed

        Returns:
            Tuple[Tensor, Tensor] (inputs, labels)
        """
        # Initialize the inputs and the labels
        inputs: Tensor = features[:, self.input_slice, :]
        labels: Tensor = features[:, self.labels_slice, :]
        labels = stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        # Return a Tuple of the inputs and the labels
        return inputs, labels









    def make_dataset(self, data: DataFrame) -> tfdata.Dataset:
        """Converts a DataFrame into a Dataset.

        Args:
            data: DataFrame
                The data to be converted into a TF Dataset
        
        Returns:
            tfdata.Dataset
        """
        # Convert the DataFrame into a numpy array
        data: ndarray = array(data, dtype=float32)

        # Initialize the Dataset
        ds: tfdata.Dataset = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=self.shuffle_data,
            batch_size=self.batch_size
        )

        # Split the Dataset into windows
        return ds.map(self.split_window)








    def __repr__(self):
        """Outputs a summary of the window.
        """
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
