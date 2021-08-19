import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATASETS_DIR = '.\\stock_dataset\\datasets'
DATASETS_FILE_NAMES = {

    "stocks_2006_2018": "all_stocks_2006-01-01_to_2018-01-01.csv"

}

CHK_NONE = 0
CHK_DROP = 1
CHK_FILL = 2

PRE_NORMALIZE = 1
PRE_INCLUDE_DIFF_HIGH_LOW = 2
PRE_INCLUDE_DIFF_CLOSE_OPEN = 4
PRE_INCLUDE_LR = 8
PRE_INCLUDE_TI = 16


def load_dataset(dataset_name=list(DATASETS_FILE_NAMES.keys())[0],  stock_name=None, split_year_train_test=None):

    dataset_file_name = DATASETS_FILE_NAMES[dataset_name]
    dataset = pd.read_csv(os.path.join(DATASETS_DIR, dataset_file_name), index_col='Date', parse_dates=['Date'])

    if stock_name is not None:

        dataset = dataset[dataset["Name"] == stock_name]

    if split_year_train_test is not None:

        train, test = dataset[:split_year_train_test], dataset[str(int(split_year_train_test) + 1):]
        return train, test, dataset

    else:

        return dataset


def plot_stock(dataset, stock_name, variable_name="Open", split=(3, 1, 1)):

    start_year = dataset.index[0].year
    end_year = dataset.index[-1].year + 1
    dataset = dataset[dataset["Name"] == stock_name]

    dataset[variable_name].plot(figsize=(16, 4), legend=True)

    plt.legend()
    plt.title('{} Stock price'.format(stock_name))
    plt.show()

    train = split[0]
    val = split[1]
    test = split[2]

    for year in range(start_year, end_year - train - val - test + 1):

        plt.figure()

        dataset[str(year):str(year + train - 1)][variable_name].plot(figsize=(16, 4), legend=True)
        dataset[str(year + train):str(year + train + val - 1)][variable_name].plot(figsize=(16, 4), legend=True)
        dataset[str(year + train + val):str(year + train + val + test - 1)][variable_name].plot(figsize=(16, 4), legend=True)

        plt.legend()
        plt.title('{} Stock price - split {} / {} / {}'.format(stock_name, year, year + train, year + train + val))
        plt.show()


def check_dataset(dataset):

    open_dataset = dataset["Open"].values
    close_dataset = dataset["Close"].values
    low_dataset = dataset["Low"].values
    high_dataset = dataset["High"].values

    print("Open values length: " + str(len(open_dataset)))
    print("Close values length: " + str(len(close_dataset)))
    print("Low values length: " + str(len(low_dataset)))
    print("High values length: " + str(len(high_dataset)))

    print()
    high_low_check_dataset = (high_dataset >= low_dataset)
    print("[High low check] - Following rows have problems: ")
    print(dataset[[not x for x in high_low_check_dataset]])
    high_low_check_dataset = np.all(high_low_check_dataset)

    print()
    high_open_check_dataset = (high_dataset >= open_dataset)
    print("[High open check] - Following rows have problems: ")
    print(dataset[[not x for x in high_open_check_dataset]])
    high_open_check_dataset = np.all(high_open_check_dataset)

    print()
    high_close_check_dataset = (high_dataset >= close_dataset)
    print("[High close check] - Following rows have problems: ")
    print(dataset[[not x for x in high_close_check_dataset]])
    high_close_check_dataset = np.all(high_close_check_dataset)

    print()
    low_open_check_dataset = (low_dataset <= open_dataset)
    print("[Low open check] - Following rows have problems: ")
    print(dataset[[not x for x in low_open_check_dataset]])
    low_open_check_dataset = np.all(low_open_check_dataset)

    print()
    low_close_check_dataset = (low_dataset <= close_dataset)
    print("[Low close check] - Following rows have problems: ")
    print(dataset[[not x for x in low_close_check_dataset]])
    low_close_check_dataset = np.all(low_close_check_dataset)
    print()

    high_low_check = high_low_check_dataset
    high_open_check = high_open_check_dataset
    high_close_check = high_close_check_dataset
    low_open_check = low_open_check_dataset
    low_close_check = low_close_check_dataset
    final_check = high_low_check and high_open_check and high_close_check and low_open_check and low_close_check

    print("Dataset correct: " + str(final_check))
    print("High low:" + str(high_low_check))
    print("High open:" + str(high_open_check))
    print("High close:" + str(high_close_check))
    print("Low open:" + str(low_open_check))
    print("Low close:" + str(low_close_check))

    return final_check


def repair_dataset(dataset, nafix=CHK_NONE):

    if nafix == CHK_DROP:

        dataset = dataset.dropna()

    elif nafix == CHK_FILL:

        dataset = dataset.fillna(method='ffill').fillna(method='bfill')

    return dataset


def get_sequences(arr, window, padding=[]):

    if len(padding) > 0:

        arr = np.vstack((padding, arr))

    y = arr[window:, 0]

    shape = (arr.shape[0] - window, window, arr.shape[1])
    strides = arr.strides[:-1] + arr.strides
    x = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    return x, y


def pre_processing(dataset, rem_features=[], lookback=None, split=(3, 1, 1), options=0, label='Close'):

    global sc
    n = 10  # window
    EPS = np.finfo('float32').eps

    def ema(arr, window):

        alpha = 2 / (window + 1)
        sma = np.average(arr[:window])

        ema = np.zeros(np.size(arr))
        ema[0] = arr[0] * alpha + sma * (1 - alpha)

        for i in range(1, len(arr)):

            ema[i] = arr[i] * alpha + ema[i - 1] * (1 - alpha)

        return ema

    Ot = dataset["Open"].values
    Ht = dataset["High"].values
    Ct = dataset["Close"].values
    Lt = dataset["Low"].values
    V = dataset["Volume"].values
    HH_n = dataset["High"].rolling(window=n, min_periods=1).max()
    LL_n = dataset["Low"].rolling(window=n, min_periods=1).min()

    if PRE_INCLUDE_TI & options:

        # EMA
        dataset["EMA"] = ema(Ct, n)

        # STOCHASTIC
        dataset["Stochastic"] = ((Ct - LL_n) / (HH_n - LL_n + EPS)) * 100

        # ROC
        close_n = np.roll(Ct, n)
        close_n[0:n] = close_n[n]
        dataset["ROC"] = ((Ct - close_n) / (close_n + EPS)) * 100

        # RSI
        sum_gain = dataset["Close"].diff(periods=-1).rolling(window=n, min_periods=1).apply(func=lambda x: np.sum(x[x < 0]), raw=True) * -1
        sum_loss = dataset["Close"].diff(periods=-1).rolling(window=n, min_periods=1).apply(func=lambda x: np.sum(x[x >= 0]), raw=True)

        dataset["RSI"] = 100 - (100 / (1 + (sum_gain / (sum_loss + EPS))))

        # AccDO
        dataset["AccDO"] = (((Ct - LL_n) - (HH_n - Ct)) / (HH_n - LL_n + EPS)) * V

        # MACD
        ema12 = ema(Ct, 12)
        ema26 = ema(Ct, 26)
        dataset["MACD"] = ema12 - ema26

        # Williams
        dataset["Williams"] = ((HH_n - Ct) / (HH_n - LL_n + EPS)) * 100

        # Disparity 5
        sma = dataset["Close"].rolling(window=5, min_periods=1).mean()
        Disp5 = (Ct / (sma + EPS)) * 100
        dataset["Disp5"] = Disp5

        # Disparity 10
        sma = dataset["Close"].rolling(window=10, min_periods=1).mean()
        Disp10 = (Ct / (sma + EPS)) * 100
        dataset["Disp10"] = Disp10

    if PRE_INCLUDE_LR & options:

        lr = ((Ct - Ot) / Ot) * 100
        dataset["LR"] = lr

    if PRE_INCLUDE_DIFF_HIGH_LOW & options:

        diff_minmax_train = Ht - Lt
        dataset["DiffHighLow"] = diff_minmax_train

    if PRE_INCLUDE_DIFF_CLOSE_OPEN & options:

        diff_minmax_train = Ct - Ot
        dataset["DiffCloseOpen"] = diff_minmax_train

    if "Name" not in rem_features:
        rem_features.append("Name")

    for feature in rem_features:

        if feature != label:

            del dataset[feature]

        else:

            print("Warning!\n Cannot delete label {} column.".format(label))

    y = dataset[label]
    del dataset[label]
    dataset.insert(0, label, y)

    start_year = dataset.index[0].year
    end_year = dataset.index[-1].year + 1

    train = split[0]
    val = split[1]
    test = split[2]

    walks = {}
    n_walks = 0

    for year in range(start_year, end_year - train - val - test + 1):

        walks['WALK_{}'.format(n_walks)] = {}

        walks['WALK_{}'.format(n_walks)]['TRAIN'] = dataset[str(year):str(year + train - 1)]
        walks['WALK_{}'.format(n_walks)]['VALIDATION'] = dataset[str(year + train):str(year + train + val - 1)]
        walks['WALK_{}'.format(n_walks)]['TEST'] = dataset[str(year + train + val):str(year + train + val + test - 1)]

        n_walks += 1

    walks['N_WALKS'] = n_walks

    if PRE_NORMALIZE & options:

        for walk in range(n_walks):

            train = walks['WALK_{}'.format(walk)]['TRAIN']
            validation = walks['WALK_{}'.format(walk)]['VALIDATION']
            test = walks['WALK_{}'.format(walk)]['TEST']

            train_stats = train.describe()
            walks['WALK_{}'.format(walk)]['STD_PARAMS'] = {}
            walks['WALK_{}'.format(walk)]['STD_PARAMS']['MEAN'] = train_stats.iloc[1, :].values
            walks['WALK_{}'.format(walk)]['STD_PARAMS']['STD'] = train_stats.iloc[2, :].values
            walks['WALK_{}'.format(walk)]['STD_PARAMS']['MIN'] = train_stats.iloc[3, :].values
            walks['WALK_{}'.format(walk)]['STD_PARAMS']['MAX'] = train_stats.iloc[7, :].values

            train = (train.values - train_stats.iloc[1, :].values) / train_stats.iloc[2, :].values
            walks['WALK_{}'.format(walk)]['TRAIN'] = get_sequences(arr=train, window=lookback)
            padding = train[-lookback:]

            validation = (validation.values - train_stats.iloc[1, :].values) / train_stats.iloc[2, :].values
            walks['WALK_{}'.format(walk)]['VALIDATION'] = get_sequences(arr=validation, window=lookback, padding=padding)
            padding = validation[-lookback:]

            test = (test.values - train_stats.iloc[1, :].values) / train_stats.iloc[2, :].values
            walks['WALK_{}'.format(walk)]['TEST'] = get_sequences(arr=test, window=lookback, padding=padding)

    else:

        for walk in range(n_walks):

            train = walks['WALK_{}'.format(walk)]['TRAIN'].values
            walks['WALK_{}'.format(walk)]['TRAIN'] = get_sequences(arr=train, window=lookback)
            padding = train[:-lookback]

            validation = walks['WALK_{}'.format(walk)]['VALIDATION'].values
            walks['WALK_{}'.format(walk)]['VALIDATION'] = get_sequences(arr=validation, window=lookback, padding=padding)
            padding = validation[:-lookback]

            test = walks['WALK_{}'.format(walk)]['TEST'].values
            walks['WALK_{}'.format(walk)]['TEST'] = get_sequences(arr=test, window=lookback, padding=padding)

    return walks


def mdd(arr, window=252):

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    roll_max = arr.rolling(window=window, min_periods=1).max()
    daily_drawdown = arr / roll_max

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    max_daily_drawdown = daily_drawdown.rolling(window=window, min_periods=1).min()
    return daily_drawdown, max_daily_drawdown

# def get_best_model(models, x_train_set, y_train_set, x_valid_set, y_valid_set, x_test_set, y_test_set, n_walk):
#     prova = 0
#     global decrease
#     decrease = False
#
#     def lr_scheduler(epoch, lr):
#         global decrease
#
#         decay_rate = 0.85
#         decay_step = 1
#
#         if prova % decay_step == 0 and decrease:
#             decrease = False
#             return lr * pow(decay_rate, np.floor(prova / decay_step))
#
#         return lr
#
#     callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
#
#     result = []
#     history_collection = []
#     min_rmse = 99999
#
#     print("Number of models:", len(models))
#     tot_time = time.time()
#     # train all models
#     for model in models:
#         print("\n***************************\n", model)
#
#         regressor = create_nn(input_shape=(x_train_set[0].shape[1], x_train_set[0].shape[2]),
#                               version=model['model'],
#                               dense_layers=model['dense_layers'],
#                               conv_units=model['conv_units'],
#                               lstm_units=model['lstm_units'],
#                               learning_rate=model['lr'])
#         history = []
#         model_time = time.time()
#         # walk forward
#         for j in range(5):
#             t = time.time()
#             for i in range(n_walk):
#                 print("\nwalk:", i, "\nSuperEpoch:", prova, "\n")
#                 history.append(
#                     regressor.fit(x_train_set[i], y_train_set[i], epochs=model['epochs'], batch_size=model['batch'],
#                                   validation_data=(x_valid_set[i], y_valid_set[i]), callbacks=callbacks))
#             prova += 1
#             decrease = True
#             t2 = time.time() - t
#             print(round(t2 / 60, 0), "min superEpoch elapsed")
#             print(round(t2 * (5 - j) / 60, 0), "min missing")
#         prova = 0
#         print(round((time.time() - model_time) / 60, 0), "min elapsed model")
#
#         history_collection.append(history)
#
#         # Prediction
#         rmse = 0
#         mape = 0
#         for i in range(n_walk):
#             prediction = regressor.predict(x_valid_set[i])
#             print(i, prediction.shape, x_valid_set[i].shape)
#             prediction = sc.inverse_transform(
#                 np.hstack((prediction, np.zeros((len(x_valid_set[i]), x_valid_set[0].shape[2] - 1)))))
#             # prediction = sc.inverse_transform(prediction)
#
#             y = y_valid_set[i].reshape(-1, 1)
#             y = sc.inverse_transform(np.hstack((y, np.zeros((len(y), x_valid_set[0].shape[2] - 1)))))
#             # y_test = sc.inverse_transform(y_test[0])
#
#             # Evaluating our model
#             plot_prediction(name, y[:, 0], prediction[:, 0])
#             r, m = return_rmse(y[:, 0], prediction[:, 0])
#             rmse += r
#             mape += m
#
#         print("\nTOT:")
#         print("RMSE: {}.".format(rmse / n_walk))
#         print("MAPE: {}.".format(mape / n_walk))
#         result.append((rmse, mape))
#
#         if rmse < min_rmse:
#             min_rmse = rmse
#             best_regressor = regressor
#
#     print(round((time.time() - tot_time) / 60, 0), "total")
#     return result, history_collection, best_regressor


def return_rmse(test, predicted):

    rmse = math.sqrt(mean_squared_error(test, predicted))
    mape = np.mean(abs((test - predicted) / test))

    print("RMSE: {}.".format(rmse))
    print("MAPE: {}.".format(mape))

    return rmse, mape

# name = 'IBM'
# split_year = '2016'
# lookback = 60
# rem_features = ["High", "Low", "Close", "Volume"]
# pre_processing_options = [NORMALIZE, INCLUDE_TI, INCLUDE_LR, INCLUDE_DIFF_OPEN_CLOSE, INCLUDE_DIFF_HIGH_LOW]
#
# options = 0
# for opt in pre_processing_options:
#     options = options | opt
#
# X_train, X_test, dataset = load_dataset(name, split_year)
# plot_stock(X_train, X_test, name, split_year)
#
# dates = dataset.index
# # X_train, X_test, y_train, y_test = pre_processing(X_train, X_test, rem_features, lookback, options = options)
#
# dataset = pre_processing_full(dataset, rem_features, lookback, options=options)
# train_sets, validation_sets, test_sets, n_walk = walk_forward(dataset, dates, 3, 1, 1)




# X_train = []
# y_train = []
# X_validation = []
# y_validation = []
# X_test = []
# y_test = []
#
# val_pred = 0
# for i in range(n_walk):
#
#     train_seq = get_sequences(train_sets[i], lookback)
#     X_train.append(train_seq[:-1])
#     y_train.append(train_seq[1:, 0, val_pred])  # get label from first element of next sequence
#
#     validation_seq = get_sequences(validation_sets[i], lookback, padding=train_seq[-1])
#     X_validation.append(validation_seq[:-1])
#     y_validation.append(validation_seq[1:, 0, val_pred])
#
#     test_seq = get_sequences(test_sets[i], lookback, padding=validation_seq[-1])
#     X_test.append(test_seq[:-1])
#     y_test.append(test_seq[1:, 0, val_pred])
#     # y_test.append(np.roll(test_seq[:,0,val_pred],1)[1:])
#
# X_train, y_train = np.asarray(X_train), np.asarray(y_train)
# X_validation, y_validation = np.asarray(X_validation), np.asarray(y_validation)
# X_test, y_test = np.asarray(X_test), np.asarray(y_test)
#
# print("X_train: {} - y_train: {}".format(X_train[0].shape, y_train[0].shape))
# print("X_validation: {} - y_validation: {}".format(X_validation[0].shape, y_validation[0].shape))
# print("X_test: {} - y_test: {}".format(X_test[0].shape, y_test[0].shape))
#
# models = get_grid_search(models=[1, 2, 3, 4], solvers=['adam'], lrs=[0.001, 0.005], epochs=[25], batches=[16, 32],
#                          lstm_units=[50, 100], dense_layers=[2], conv_units=[128])
# get_best_model(models, X_train, y_train, X_validation, y_validation, X_test, y_test, n_walk)