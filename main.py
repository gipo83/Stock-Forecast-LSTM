from stock_dataset import stock_dataset
from stock_dataset.stock_regressors import ModelFactory
import numpy as np


# [ MAIN ]
def main():

    stock_name = 'IBM'
    rem_features = ["High", "Low", "Volume", "Open", "Close"]
    lookback = 15
    split = (3, 1, 1)
    high_low = (-1, 1)
    pre_processing_options = [stock_dataset.PRE_NORMALIZE,
                              stock_dataset.PRE_INCLUDE_LR]

    norm_options = {

        "METHOD": stock_dataset.NORM_Z_SCORE,
        "HIGH_LOW": high_low

    }

    # options = 0
    # for opt in pre_processing_options:
    #      options = options | opt
    #
    # dataset = stock_dataset.load_dataset(stock_name=stock_name)
    # stock_dataset.check_dataset(dataset=dataset)
    # dataset = stock_dataset.repair_dataset(dataset=dataset, nafix=stock_dataset.CHK_FILL)
    #
    # stock_dataset.plot_stock(dataset=dataset, stock_name=stock_name, variable_name='Open', split=(3, 1, 1))

    # dataset = stock_dataset.pre_processing(dataset=dataset, rem_features=rem_features, lookback=lookback, split=(3, 1, 1), options=options, label='LR')

    mf = ModelFactory(rem_features=rem_features, lookback=lookback, split=split, options=pre_processing_options, label="LR", norm_options=norm_options)
    mf.add_grid_search(models=[1], epochs=[50], batches=[16], learning_rates=[0.001], learning_rate_steps=[5], learning_rate_decays=[0.90], dense_layers=[3], lstm_units=[50])
    mf.grid_search()

    return 0


# [ RUN ]
if __name__ == '__main__':

    main()
