from stock_dataset import stock_dataset
from stock_dataset.stock_regressors import ModelFactory
from stock_dataset.stock_regressors import guessing_test
from stock_dataset.stock_regressors import metrics
import numpy as np
import pandas as pd


# [ MAIN ]
def main():

    stock_name = 'IBM'
    rem_features = ["High", "Low", "Volume", "Open", "Close"]
    lookback = 60
    split = (3, 1, 1)
    high_low = (-1, 1)

    pre_processing_options = [stock_dataset.PRE_INCLUDE_TI,
                              stock_dataset.PRE_INCLUDE_LR,
                              stock_dataset.PRE_NORMALIZE]

    norm_options = {

        "METHOD": stock_dataset.NORM_Z_SCORE,
        "HIGH_LOW": high_low,
        "ORDER": 1

    }

    # options = 0
    # for opt in pre_processing_options:
    #       options = options | opt
    #
    # dataset = stock_dataset.load_dataset(stock_name=stock_name)
    # dataset = stock_dataset.repair_dataset(dataset=dataset, nafix=stock_dataset.CHK_FILL)
    # stock_dataset.check_dataset(dataset=dataset)
    #
    # stock_dataset.plot_stock(dataset=dataset, stock_name=stock_name, variable_name='Open', split=(3, 1, 1))
    #
    # dataset = stock_dataset.pre_processing(dataset=dataset, rem_features=rem_features, lookback=lookback, split=(3, 1, 1), options=options, label='Close')
    #
    # print(dataset['WALK_0']['TRAIN'][1].shape)
    # print(dataset['WALK_0']['VALIDATION'][1].shape)
    # print(dataset['WALK_0']['TEST'][1].shape)

    mf = ModelFactory(rem_features=rem_features, lookback=lookback, split=split, options=pre_processing_options, label="LR", norm_options=norm_options)
    #mf.add_grid_search(models=[2], epochs=[70], batches=[16, 32], learning_rates=[0.01, 0.001], learning_rate_steps=[10, 5], learning_rate_decays=[0.90], dense_layers=[1, 2], lstm_units=[64, 128])
    mf.add_grid_search(models=[2], epochs=[70], batches=[32], learning_rates=[0.01], learning_rate_steps=[5], learning_rate_decays=[0.90], dense_layers=[1], lstm_units=[64])
    mf.grid_search(result_path='./grid_search_results_70')

    data = "TEST"
    mf.evaluate(data=data)
    for i in range(mf.walks["N_WALKS"]):
        if i == 0:
            y, x = mf.denormalize_pred_walk(walk=i, data=data)
        else:
            tmp_y, tmp_x = mf.denormalize_pred_walk(walk=i, data=data)
            y = np.hstack((y, tmp_y))
            x = np.hstack((x, tmp_x))

    met = metrics(y, x)
    print(met["RMSE"].values)
    print(met["MAPE"].values)

    offset = 0
    guessing_test(x, y, offset)

    return 0


# [ RUN ]
if __name__ == '__main__':

    main()
