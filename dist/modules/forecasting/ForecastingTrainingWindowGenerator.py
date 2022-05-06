from typing import Union, List, Tuple
from pandas import DataFrame
from numpy import arange, ndarray, array, float32
from tensorflow import stack, Tensor, data as tfdata
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from modules.forecasting import ITrainingWindowGeneratorConfig





class ForecastingTrainingWindowGenerator:
    """ForecastingTrainingWindowGenerator Class

    This class manages the data windowing for the train, validation and test data.

    Class Properties:
        ...

    Instance Properties:
        train_df: DataFrame
            ...
        val_df: DataFrame
            ...
        test_df: DataFrame
            ...
        label_columns: Union[List[str], None]
            ...
        column_indices: ...
            ...
        input_width: int
            ...
        label_width: int
            ...
        shift: int
            ...
        total_window_size: int
            ...
        input_slice: slice
            ...
        input_indices: ndarray
            ...
        label_start: int
            ...
        labels_slice: slice
            ...
        label_indices: ndarray
            ...
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
        self.label_columns: Union[List[str], None] = config.get('label_columns')
        if self.label_columns is not None:
            self.label_columns_indices = { name: i for i, name in enumerate(self.label_columns) }
        self.column_indices = { name: i for i, name in enumerate(self.train_df.columns) }

        # Work out the window parameters.
        self.input_width: int = config['input_width']
        self.label_width: int = config['label_width']
        self.shift: int = config['shift']

        # Calculate the total window size
        self.total_window_size: int = self.input_width + self.shift

        # Initialize the Input Data
        self.input_slice: slice = slice(0, self.input_width)
        self.input_indices: ndarray = arange(self.total_window_size)[self.input_slice]

        # Initialize the Label Data
        self.label_start: int = self.total_window_size - self.label_width
        self.labels_slice: slice = slice(self.label_start, None)
        self.label_indices: ndarray = arange(self.total_window_size)[self.labels_slice]





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
        if self.label_columns is not None:
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
            shuffle=True,
            batch_size=32
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
