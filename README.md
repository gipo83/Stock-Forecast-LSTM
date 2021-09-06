# Stock-Forecast-LSTM
The goal of this project is generating a LSTM neural network with Keras able to predict stock prices based on past data only.

We consider the input dataset with these features: Open, Low, High, Close, Volume, Name.

First of all download the project and import library:
```
import stockForecastLSTM
```

Some informations about preprocessing are:
  - `remove features`: list of the features that you want remove before training;
  - `lookback`: sequencies' length;
  - `split`: tuple with 3 value that indicate the length in years of training set, validation set and test set;
  - `preprocessing options`: constant values to set the desired operations of preprocessing;
    - `PRE_NORMALIZE`: normalization of data;
    - `PRE_INCLUDE_LR`: add the 10 predecessor LR value for each row;
    - `PRE_INCLUDE_TI`: add technical indicators (EMA, Stochastic, ROC, RSI, AccDO, MACD, Williams, Disparity 5, Disparity 10);
  - `normalization options`: values to set a king of normalization in preprocessing step (it will be executed only whether in preprocessing options is requested);
    - `METHOD`: method of normalization (NORM_MIN_MAX or NORM_Z_SCORE);
    - `HIGH_LOW`: tuple for min max normalization;
  - `stock name`: list of stocks that you want include in training;
  - `label`: value to predict.

We show an example below
```
rem_features = ["High", "Low", "Volume", "Open","Close"]
lookback = 60
split = (3, 1, 1)
high_low = (0.001, 1)
pre_processing_options = [PRE_NORMALIZE,
                          PRE_INCLUDE_LR,
                          PRE_INCLUDE_TI]

norm_options = {
    "METHOD": NORM_MIN_MAX, #NORM_Z_SCORE,
    "HIGH_LOW": high_low
}

stock_name = ["IBM"]
label = "IBM_LR" #"IBM_Close"
```

You can make the instance of object with this line of code:
```
mmf = MultiModelFactory(stock_name_list = stock_name,
                        rem_features = rem_features, 
                        lookback = lookback, 
                        split = split, 
                        options = pre_processing_options, 
                        label = label, 
                        norm_options = norm_options)
```

Choice the values that you want change in grid search with the follow code:
```
mmf.add_grid_search(models = [1], 
                    epochs = [50, 70], 
                    batches = [16, 32], 
                    learning_rates = [0.001, 0.005], 
                    learning_rate_steps = [10], 
                    learning_rate_decays = [0.90], 
                    dense_layers = [1],                 # number of dense layer added at the end of net before the last one
                    lstm_units = [64])                  # number of units in all lstm layers in the net
```
The number of model indicates the model in our list of model tested (1 is the better).

If you want add other test in grid search you can: 
```
mmf.add_grid_search(models = [2], 
                    epochs = [50], 
                    batches = [64], 
                    learning_rates = [0.01], 
                    learning_rate_steps = [10], 
                    learning_rate_decays = [0.90], 
                    dense_layers = [1], 
                    lstm_units = [64])
```

To start with training you just have to:
```
mmf.grid_search(data='VALIDATION')
```
