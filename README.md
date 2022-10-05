# EPOCH BUILDER

The Epoch Builder is a cluster of machines designed to train and evaluate many Prediction Models. It also outputs an Epoch File that can be installed in Epoca's Platform.


#
## Requirements

- Python: v3.8.10

- Pip: v20.0.2

The dependencies are located in the **requirements.txt** file and can be installed with:

`pip3 install -r requirements.txt`


### Cluster Manager Launcher Requirements

- NodeJS: v16.14.0

- NPM: 8.3.1

The dependencies in the **package.json** file can be installed with:

`npm install`

#
## Structure

```
epoch-builder
    │
    _EPOCH_NAME/
    ├───... <- Epoch Assets
    │
    candlesticks/
    ├───... <- Processed candlestick's bundle
    │
    config/
    ├───... <- Global configurations
    │
    dist/
    ├───cluster/
    │   ├──... <- Cluster Manager's source code (pure js)
    │   modules/
    │   ├──... <- Epoch Builder's source code
    │   tests/
    │   ├──... <- Epoch Builder's unit tests
    │   │
    │   some_script_endpoint.py
    │
    package.json <- Cluster Manager's dependencies
    │
    requirements.txt <- Epoch Builder's dependencies
```


#
## Cluster Manager Launcher

The launcher initializes a CLI that manages the cluster. It can be started with:

`npm start`




#
## Getting Started

Generate a fresh Candlestick Bundle from the GUI and **decompress it** in the root directory.

```
epoch-builder
    candlesticks/
    ├───candlesticks.csv <- 1m interval
    └───prediction_candlesticks.csv <- 30m interval
```

Once the Epoch is created, the candlestick files will be adjusted to the epoch's width and a the normalized variation of the 30m candlesticks will also be placed in the directory.

```
epoch-builder
    candlesticks/
    ├───candlesticks.csv
    ├───prediction_candlesticks.csv
    └───normalized_prediction_candlesticks.csv
```

The Epoch's configuration file is stored in the root configuration directory:

```
epoch-builder
    config/
    ├───cluster.json
    └───epoch.json
```

Finally, the directory and subdirectories that hold the Epoch's assets are also created:

```
epoch-builder
    _EPOCH_NAME/
    └───prediction_models/
        ├──...
        regression_batched_certificates/
        ├──...
        regression_training_configs/
        ├──...
        regressions/
        ├──...
        _EPOCH_NAME_receipt.txt
```


#
## Unit Tests

In order to be able to run the unit tests, the regression configurations must be generated and the unit test model must be trained. 

It is also important to mention that even though the unit tests can be executed in any machine within the cluster, they are automatically executed on the localhost machine.


#
## Regression Factory

### Configurations

In order to build the regressions that will be used by the prediction model, the configurations must be generated. This process is also known as Hyperparameter Tuning.

The training configurations are saved in batches by category, following the structure:

```
epoch-builder
    _EPOCH_NAME/
    └───regression_training_configs/
        ├──CDNN/
        │  ├───KR_CDNN_1_10.json
        │  └───...
        ├──CLSTM/
        │  └───...
        ├──DNN/
        │  └───...
        ├──LSTM/
        │  └───...
        ├──UNIT_TEST/
        │  └───...
        receipt.txt
```

### Training

Once the regression configurations have been generated and the unit tests are passing, the training process begins.

One machine can train one batch at a time. It is important to keep track of the training progress in a platform such as Trello.

When a single Regression is trained, the model's file and the training certificate are saved as follows:

```
epoch-builder
    _EPOCH_NAME/
    └───regressions/
        └──KR_UNIT_TEST/
           ├───certificate.json
           └───model.h5
```

Moreover, when a full batch is trained, a combined certificate is stored in order to be able to visualize many certificates simultaneously through the GUI.

```
epoch-builder
    _EPOCH_NAME/
    └───regression_batched_certificates/
        ├──KR_UNIT_TEST.json
        └──...
```



#
## Prediction Model Factory


### Initialization

Once the regressions have gone through the training process and the top 20 have been selected, the prediction model can be initialized. During this process, the prediction model assets and configurations are generated:

#### Assets

1) **features:** lists of all the features by regression.

2) **labels:** the list of outcomes by price change requirement.

3) **lookback_indexer:** A dict containing the prediction candlestick indexes mapped to 1m candlestick open times.

#### Configurations

In order to find the most profitable model for the test dataset, a series of configurations are generated in order to cover as many alternatives as possible by making use of Hyperparameter Tuning Techniques.

An initialized prediction model has the following structure:

```
epoch-builder
    _EPOCH_NAME/
    └───prediction_models/
        ├──assets/
        │  ├──features.json
        │  ├──labels.json
        │  └──lookback_indexer.json
        ├──configs/
        │  ├──_EPOCH_NAME_1_87.json
        │  └──...
        ├──profitable_configs/ <- Empty
        configs_receipt.txt
```



### Profitable Configurations

As profitable configurations are found, they are placed in the **profitable_configs** directory which is then read in order to generate the Prediction Model Build:

```
epoch-builder
    _EPOCH_NAME/
    └───prediction_models/
        └──build.json
```




#
## Epoch Export

Once the best prediction model is found, the Epoch Builder gathers all the neccessary assets and builds the **Epoch File** which then can be installed in Epoca. 

```
epoch-builder
    _EPOCH_NAME/
    └───_EPOCH_NAME.zip
```

Finally, the candlesticks directory (**candlesticks/**) and the epoch's configuration file (**config/epoch.json**) are placed in the root of the Epoch's directory (**_EPOCH_NAME/**) in order to make it archivable whilst maintining reproducibility.






#
## Cluster Manager

The cluster manager can perform actions on localhost or any machine within the cluster. The actions are divided in the following categories:


### Server

`connect_to_a_server:` Opens a SSH connection with a server.

`view_server_status:` Displays details about the server's resources as well as the running process.

`subscribe_to_server_logs:` Creates a persistant read on the file that holds the process' logs.

`reboot_server:` Resets a server (as sudo).

`shutdown_server:` Turns off a server (as sudo)

`kill_process:` Kills any python3 process that is running.

`install_ssh_key_on_a_server:` Installs the SSH on a server in order to avoid entering the password per action.


### Epoch Builder

`create_epoch:` Creates a brand new epoch.

`generate_regression_training_configs:` Generates all the regression training configurations (Hyperparameter tuning).

`train_regression_batch:` Runs the regression training on a selected configuration batch.

`initialize_prediction_models:` Creates all the prediction models' assets as well as the configurations.

`find_profitable_configs:` Finds all the prediction model profitable configurations in a batch and then stores them separately.

`build_prediction_models:` Builds the prediction models that were found profitable.

`export_epoch:` Exports the selected Prediction Model as well as all the required assets in the Epoch File.

`unit_tests:` Runs the end-to-end tests on the localhost machine.


### Push

`push_root_files:` Pushes any root files such as the requirements.txt, package.json, etc.

`push_configuration:` Pushes the root configuration directory.

`push_candlesticks:` Pushes the candlesticks directory.

`push_dist:` Pushes the entire distribution directory. It contains modules, tests and script endpoints.

`push_regression_training_configs:` Pushes the regression training configurations directory.

`push_prediction_models:` Pushes the prediction models directory. It contains the assets and the configurations.

`push_epoch_builder:` Performs a full push of the Epoch Builder.


### Pull

`pull_trained_regressions:` Pulls the model files and certificates generated from a training batch. It also pulls the combined certificates.

`pull_prediction_models:` Pulls the profitable prediction model configurations.





#
## Candlestick Information

A candlestick is an object comprised by 7 properties that describe the price movements during a specific period of time. 

Candlesticks can be represented in intervals of 1, 15, 30 or even 60 minutes. Epoca makes use of the 1 minute interval candlesticks for evaluating backtests, trading sessions and simulations as well as 30 minute interval candlesticks for training models and generating predictions.


### Anatomy

| Name | Alias | Type | Description
| ---- | ----- | ---- | -----------
| Open Time | ot | int | The timestamp (in ms) in which the candlestick came into existence. Should be equal to the previous candlestick’s close time plus 1 millisecond.
| Close Time | ct | int | The timestamp (in ms) in which the candlestick ended. Should be equal to the next candlestick’s open time minus 1 millisecond.
| Open Price | o | float | Open Price in USDT (Should be equals or very close to the previous candlestick’s close price)
| High Price | h | float | Highest Price in USDT traded during the candlestick
| Low Price | l | float | Lowest Price in USDT traded during the candlestick
| Close Price | c | float | Close Price in USDT (Should be equals or very close to the next candlestick’s open price)
| Volume | v | float | Volume in USDT traded during the candlestick



### DataFrame Example

|  | ot | ct | o | h | l | c | v
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1502942400000 | 1502944199999 | 4261.48 | 4280.56 | 4261.32 | 4261.45 | 48224.70
| 1 | 1502944200000 | 1502945999999 | 4280.00 | 4313.62 | 4267.99 | 4308.83 | 154141.27
| 2 | 1502946000000 | 1502947799999 | 4308.83 | 4328.69 | 4304.31 | 4320.00  | 90864.17
