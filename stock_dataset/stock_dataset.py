import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import stock_dataset.stock_dataset

DATASETS_DIR = 'stock_dataset'
DATASETS_SUB_DIR = 'datasets'

DATASETS_FILE_NAMES = {

    "stocks_2006_2018": "all_stocks_2006-01-01_to_2018-01-01.csv"

}

CHK_NONE = 0
CHK_DROP = 1
CHK_FILL = 2

PRE_NORMALIZE = 1
PRE_DIFFERENCING = 2
PRE_INCLUDE_DIFF_HIGH_LOW = 4
PRE_INCLUDE_DIFF_CLOSE_OPEN = 8
PRE_INCLUDE_LR = 16
PRE_INCLUDE_TI = 32

NORM_MIN_MAX = 0
NORM_Z_SCORE = 1

DIFF_BEFORE = 0
DIFF_AFTER = 1


def load_dataset(dataset_name=list(DATASETS_FILE_NAMES.keys())[0],  stock_name=None, split_year_train_test=None):

    dataset_file_name = DATASETS_FILE_NAMES[dataset_name]
    dataset = pd.read_csv(os.path.join(DATASETS_DIR, DATASETS_SUB_DIR, dataset_file_name), index_col='Date', parse_dates=['Date'])

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

    plt.figure()
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

    return final_check, high_low_check, high_open_check, high_close_check, low_open_check, low_close_check


def repair_dataset(dataset, problems= None, nafix=CHK_NONE):

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


def difx(s, order=0):

    for i in range(order):

        s = s.diff().combine_first(s)

    return s


def rev_difx(s, order=0):

    for i in range(order):

        s = s.cumsum()

    return s


def rolling_window(a, window, include_last=False):
    tmp = -1

    if include_last:
        tmp = 0

    padding = np.zeros(window - (1 + tmp))
    a = np.hstack((padding, a))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    if include_last:

        res = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    else:

        res = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[:-1]

    return res


def pre_processing(dataset, rem_features=[], lookback=None, split=(3, 1, 1), options=0, label='Close', norm_options={"METHOD": NORM_MIN_MAX, "HIGH_LOW": (0, 1), "ORDER": 0}):

    dataset = dataset.copy(deep=True)

    def ema(arr, window):

        alpha = 2 / (window + 1)
        sma = np.average(arr[:window])

        ema = np.zeros(np.size(arr))
        ema[0] = arr[0] * alpha + sma * (1 - alpha)

        for i in range(1, len(arr)):

            ema[i] = arr[i] * alpha + ema[i - 1] * (1 - alpha)

        return ema
    

    window = 10
    EPS = np.finfo('float32').eps
    
    Ot = dataset["Open"].values
    Ht = dataset["High"].values
    Ct = dataset["Close"].values
    Lt = dataset["Low"].values
    V = dataset["Volume"].values
    HH_n = dataset["High"].rolling(window=window, min_periods=1).max()
    LL_n = dataset["Low"].rolling(window=window, min_periods=1).min()

    if PRE_INCLUDE_TI & options:

        # EMA
        dataset["EMA"] = ema(Ct, window)

        # STOCHASTIC
        dataset["Stochastic"] = ((Ct - LL_n) / (HH_n - LL_n + EPS)) * 100

        # ROC
        close_n = np.roll(Ct, window)
        close_n[0:window] = close_n[window]
        dataset["ROC"] = ((Ct - close_n) / (close_n + EPS)) * 100

        # RSI
        sum_gain = dataset["Close"].diff(periods=-1).rolling(window=window, min_periods=1).apply(func=lambda x: np.sum(x[x < 0]), raw=True) * -1
        sum_loss = dataset["Close"].diff(periods=-1).rolling(window=window, min_periods=1).apply(func=lambda x: np.sum(x[x >= 0]), raw=True)

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

        lr = ((Ct - Ot) / Ot)
        dataset["LR"] = lr

        lr = rolling_window(lr, window)
        for i in range(lr.shape[1]):
            dataset["LR_{}".format(i)] = lr[:, i]

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

            print("Warning!\nCannot delete label {} column.".format(label))

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

        if PRE_DIFFERENCING & options and norm_options["DIFF_PER"] == DIFF_BEFORE:

            walks['WALK_{}'.format(n_walks)]['TRAIN'] = difx(walks['WALK_{}'.format(n_walks)]['TRAIN'], norm_options['ORDER'])
            walks['WALK_{}'.format(n_walks)]['VALIDATION'] = difx(walks['WALK_{}'.format(n_walks)]['VALIDATION'], norm_options['ORDER'])
            walks['WALK_{}'.format(n_walks)]['TEST'] = difx(walks['WALK_{}'.format(n_walks)]['TEST'], norm_options['ORDER'])

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

            if norm_options["METHOD"] == NORM_MIN_MAX:

                low = norm_options["HIGH_LOW"][0]
                high = norm_options["HIGH_LOW"][1]

                train = ((high - low) * (train.values - train_stats.iloc[3, :].values) / (train_stats.iloc[7, :].values - train_stats.iloc[3, :].values)) + low
                validation = ((high - low) * (validation.values - train_stats.iloc[3, :].values) / (train_stats.iloc[7, :].values - train_stats.iloc[3, :].values)) + low
                test = ((high - low) * (test.values - train_stats.iloc[3, :].values) / (train_stats.iloc[7, :].values - train_stats.iloc[3, :].values)) + low

            elif norm_options["METHOD"] == NORM_Z_SCORE:

                train = (train.values - train_stats.iloc[1, :].values) / train_stats.iloc[2, :].values
                validation = (validation.values - train_stats.iloc[1, :].values) / train_stats.iloc[2, :].values
                test = (test.values - train_stats.iloc[1, :].values) / train_stats.iloc[2, :].values

            if PRE_DIFFERENCING & options and norm_options["DIFF_PER"] == DIFF_AFTER:

                train = difx(pd.DataFrame(train), norm_options["ORDER"]).values
                validation = difx(pd.DataFrame(validation), norm_options["ORDER"]).values
                test = difx(pd.DataFrame(test), norm_options["ORDER"]).values

            walks['WALK_{}'.format(walk)]['TRAIN'] = get_sequences(arr=train, window=lookback)
            padding = train[-lookback:]

            walks['WALK_{}'.format(walk)]['VALIDATION'] = get_sequences(arr=validation, window=lookback, padding=padding)
            padding = validation[-lookback:]

            walks['WALK_{}'.format(walk)]['TEST'] = get_sequences(arr=test, window=lookback, padding=padding)

    else:

        for walk in range(n_walks):

            train = walks['WALK_{}'.format(walk)]['TRAIN'].values
            walks['WALK_{}'.format(walk)]['TRAIN'] = get_sequences(arr=train, window=lookback)
            padding = train[-lookback:]

            validation = walks['WALK_{}'.format(walk)]['VALIDATION'].values
            walks['WALK_{}'.format(walk)]['VALIDATION'] = get_sequences(arr=validation, window=lookback, padding=padding)
            padding = validation[-lookback:]

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
