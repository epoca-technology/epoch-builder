# PREDICTION BACKTESTING v0.0.3

Plutus' Prediction Backtesting is a project designed to experiment with different Prediction Models in order to optimize profits.


#
## Requirements

- Python: v3.8.10

- Pip: v20.0.2

- [PostgreSQL: v14.3](./docs/POSTGRES.md) (On the Plutus Tester Server)

The dependencies are located in the **requirements.txt** file and can be installed with:

`pip3 install -r requirements.txt`


#
## Structure

```
prediction-backtesting
    │
    backtest_assets/
    ├───configurations/
    │   ├──arima
    │   │  └──...
    │   └──classification
    │      └──...
    │
    ├───results/
    │   ├──BACKTEST_01_1649788629072.json
    │   └──BACKTEST_02_1650551357902.json
    │
    candlesticks/
    ├───candlesticks.csv
    ├───prediction_candlesticks.csv
    │
    config/
    ├───ArimaCombinations.json
    ├───Backtest.json
    ├───ClassificationTrainingData.json
    ├───RegressionTraining.json
    │
    db_management/
    ├───backup/
    │   └──1652662191014.dump <- Backup File Generated through the CLI
    ├───restore/
    │   └──1652495666677.dump <- Backup File to be restored
    │
    dist/
    ├───modules/
    │   ├──some_module/
    │   │  └──SomeModule.py
    │   └──some_other_module/
    │      └──SomeOtherModule.py
    │
    ├───tests/
    │   ├──some_module_test.py
    │   └──some_other_module_test.py
    │
    ├───run_arima_combinations.py
    ├───run_backtest.py
    ├───run_classification_training_data.py
    ├───run_db_management.py
    ├───run_regression_training.py
    │
    keras_assets/
    ├───batched_training_certificates/
    │   ├──REGRESSION_UNIT_TEST_1652967541197.json
    │   └──CLASSIFICATION_UNIT_TEST_1652967541197.json
    │
    ├───classification_training_data/
    │   └──fed1a436-190f-473b-8f21-ae1f2ede0734.json
    │
    ├───model_configs/
    │      ├──Classification/
    │      │  └──SomeBatchConfig.json
    │      └──Regression/
    │         └──SomeBatchConfig.json
    │
    ├───models/
    │      ├──R_UNIT_TEST/
    │      │  ├──certificate.json
    │      │  └──model.h5
    │      └──C_UNIT_TEST/
    │         ├──certificate.json
    │         └──model.h5
    │
    ArimaCombinations.sh
    │
    Backtest.sh
    │
    ClassificationTrainingData.sh
    │
    DatabaseManagement.sh
    │
    RegressionTraining.sh
    │
    UnitTests.sh
```



#
## Getting Started

- Generate the **candlesticks.csv** and **prediction_candlesticks.csv** files through the **compose** program and place them inside of the **./candlesticks** directory.

- Set the permissions on the executables (This only needs to be done once):

  `chmod u+x ArimaCombinations.sh Backtest.sh ClassificationTrainingData.sh DatabaseManagement.sh RegressionTraining.sh UnitTests.sh`






#
## Database Management

Basic utility that allows interactions with the PostgreSQL Database such as:

- Visualize the Database Summary

- Create a Database Backup

- Restore a Database Backup

In order to execute the utility run the following:

`./DatabaseManagement.sh`


### Database Backup

When the Database Backup is executed, the dump file is placed in the following path:

**db_management/backup**

### Database Restore

Before running the restore function, make sure to have placed the backup file that you wish to restore in the following path:

**db_management/restore**





#
## Arima Combinations

Arima Combinations is a script that generates **Backtest Configuration Files** and places them in the **backtest_assets/configurations** directory. Before executing the script, input the desired configuration values in **config/ArimaCombinations.json**.

Run the generator by executing the following:

`./ArimaCombinations`






#
## Backtests

Input the desired configuration values in **config/Backtest.json** and run:

`./Backtest.sh`

Once the execution completes, the results will be placed under the **./backtest_assets/results** directory in the following format: **{BACKTEST_ID}_{TIMESTAMP}.json**






#
## Regression Training

Input the desired configuration values in **config/RegressionTraining.json** and run:

`./RegressionTraining.sh`

Once the execution completes, the models and their certificates are saved in **keras_assets/models**. The certificates batch on the other hand is stored in **keras_assets/batched_training_certificates**.





#
## Classification Training Data

Input the desired configuration values in **config/ClassificationTrainingData.json** and run:

`./ClassificationTrainingData`

Once the execution completes, a file with a **uuid4 as the name** will be generated and placed in the **keras_assets/classification_training_data** directory.






#
## Unit Tests

Run an end-to-end unit test with the following command:

`./UnitTests`

For the unit tests to pass, a Postgres connection must be successfully established and the unit test keras models must be in the correct directory.







#
## Candlesticks

A candlestick is an object comprised by 7 properties that describe the price movements during a specific period of time. 

Candlesticks can be represented in intervals of 1, 15, 30 or even 60 minutes. Plutus makes use of the 1 minute interval candlesticks for evaluating trading sessions and simulations as well as 30 minute interval candlesticks for performing predictions on future prices.


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