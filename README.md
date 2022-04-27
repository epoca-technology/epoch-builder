# PREDICTION BACKTESTING

Plutus' Prediction Backtesting is a project designed to experiment with different Prediction Models in order to optimize profits.


#
## Requirements

- Python: v3.8.10

- Pip: v20.0.2

The dependencies are located in the **requirements.txt** file and can be installed with:

`pip3 install -r requirements.txt`


#
## Structure

```
prediction-backtesting
    │
    backtest_results/
    ├───BACKTEST_01_1649788629072.json
    ├───BACKTEST_02_1650551357902.json
    │
    candlesticks/
    ├───candlesticks.csv
    ├───prediction_candlesticks.csv
    │
    db/
    ├───db.sqlite
    │
    db_merge/
    ├───db1.sqlite
    ├───db2.sqlite
    ├───result.sqlite
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
    │
    plutus_tester_backtests/
    ├───COMBINATION_ID/
    │   ├──COMBINATION_ID_1.json
    │   └──COMBINATION_ID_2.json
    │
    training_data/
    ├───@TODO/
    │   ├──@TODO.json
    │   └──@TODO.json
    │
    ArimaCombinations_config.json
    ArimaCombinations.sh
    │
    Backtest_config.json
    Backtest.sh
    │
    DBMerge.sh
    │
    TrainingData_config.json
    TrainingData.sh
    │
    UnitTests.sh
```



#
## Getting Started

- Generate the **candlesticks.csv** and **prediction_candlesticks.csv** files through the **compose** program and place them inside of the **./candlesticks** directory.

- Set the permissions on the executables (This only needs to be done once):

  `chmod u+x ArimaCombinations.sh Backtest.sh DBMerge.sh TrainingData.sh UnitTests.sh`



#
## Arima Combinations

Arima Combinations is a script that generates **Backtest_config** files and places them in the **plutus_tester_backtests** directory. Before executing the script, input the desired configuration values in **ArimaCombinations_config.json**.

Run the generator by executing the following:

`./ArimaCombinations`


#
## Backtests

Input the desired configuration values in **ArimaCombinations_config.json**

Run the Backtest by executing the following:

`./Backtest.sh`

Once the execution completes, the results will be placed under the **./backtest_results** directory in the following format: **{BACKTEST_ID}_{TIMESTAMP}.json**




#
## Training Data

@TODO 




#
## DB Merge

Place the .sqlite files in the **./db_merge** directory and run the following:

`./DBMerge.sh`

Once the execution completes, the merged database file will be placed under the **./db_merge** directory with the following name: **result.sqlite**



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