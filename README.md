# PREDICTION BACKTESTING

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
    backtest_configurations/
    ├───COMBINATION_ID/
    │   ├──COMBINATION_ID_1.json
    │   └──COMBINATION_ID_2.json
    │
    backtest_results/
    ├───BACKTEST_01_1649788629072.json
    ├───BACKTEST_02_1650551357902.json
    │
    candlesticks/
    ├───candlesticks.csv
    ├───prediction_candlesticks.csv
    │
    config/
    ├───ArimaCombinations.json
    ├───Backtest.json
    ├───TrainingData.json
    │
    db/
    ├───db.sqlite
    ├───merge_result.sqlite <- Output when running ./DBMerge.sh
    │
    db_merge/
    ├───db1.sqlite
    ├───db2.sqlite
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
    ├───run_db_merge.py
    ├───run_training_data.py
    │
    training_data/
    ├───1649788629072/ <- Generated Data
    │   ├──data.csv
    │   └──receipt.json
    │
    ├───data.csv       <- Used for training a model
    ├───receipt.json   <- Used to validate the data & the model
    │
    ArimaCombinations.sh
    │
    Backtest.sh
    │
    DBMerge.sh
    │
    RegressionTraining.sh
    │
    TrainingData.sh
    │
    UnitTests.sh
```



#
## Getting Started

- Generate the **candlesticks.csv** and **prediction_candlesticks.csv** files through the **compose** program and place them inside of the **./candlesticks** directory.

- Set the permissions on the executables (This only needs to be done once):

  `chmod u+x ArimaCombinations.sh Backtest.sh DatabaseManagement.sh RegressionTraining.sh TrainingData.sh UnitTests.sh`



#
## Arima Combinations

Arima Combinations is a script that generates **Backtest Configuration Files** and places them in the **backtest_configurations** directory. Before executing the script, input the desired configuration values in **config/ArimaCombinations.json**.

Run the generator by executing the following:

`./ArimaCombinations`


#
## Backtests

Input the desired configuration values in **config/Backtest.json** and run:

`./Backtest.sh`

Once the execution completes, the results will be placed under the **./backtest_results** directory in the following format: **{BACKTEST_ID}_{TIMESTAMP}.json**




#
## Training Data

Input the desired configuration values in **config/TrainingData.json** and run:

`./TrainingData`

Once the execution completes, the files **data.csv** and **receipt.json** will be placed under the **training_data/{TIMESTAMP}** directory.




#
## DB Merge

Merges the Local DB file from **db/db.sqlite** and all the DB files located in **db_merge**. 

`./DBMerge.sh`

Once the execution completes, the result is placed in **db/merge_result.sqlite** and the files located in **db_merge** are automatically deleted.




#
## Unit Tests

Run an end-to-end unit test with the following command:

`./UnitTests`




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