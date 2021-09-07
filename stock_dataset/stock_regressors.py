import json
import os
import time
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import ParameterGrid

from stock_dataset import stock_dataset


class ModelFactory:

    def __init__(self, dataset_name=list(stock_dataset.DATASETS_FILE_NAMES.keys())[0], stock_name="IBM", rem_features=[], lookback=60, split=(3, 1, 1), options=[], label="Close", norm_options={"METHOD": stock_dataset.NORM_MIN_MAX, "HIGH_LOW": (0, 1)}):

        self.lookback = lookback
        self.split = split

        pre_processing_options = 0
        for opt in options:
            pre_processing_options = pre_processing_options | opt

        self.options = pre_processing_options
        self.label = label
        self.norm_options = norm_options

        self.stock_name = stock_name
        self.dataset = stock_dataset.load_dataset(dataset_name=dataset_name, stock_name=self.stock_name)
        self.dataset = stock_dataset.repair_dataset(dataset=self.dataset, nafix=stock_dataset.CHK_FILL)
        stock_dataset.check_dataset(self.dataset)
        self.walks = stock_dataset.pre_processing(dataset=self.dataset, rem_features=rem_features, lookback=lookback, split=split, options=self.options, label=label, norm_options=norm_options)
        self.models = []

    def add_grid_search(self, models=[1], solvers=['adam'], learning_rates=[0.001], epochs=[1], batches=[32], lstm_units=[50], dense_layers=[0], conv_units=[0], learning_rate_decays=[1], learning_rate_steps=[1]):

        grid = dict(model=models,
                    solver=solvers,
                    learning_rate=learning_rates,
                    epochs=epochs,
                    batch=batches,
                    lstm_units=lstm_units,
                    dense_layers=dense_layers,
                    conv_units=conv_units,
                    learning_rate_decay=learning_rate_decays,
                    learning_rate_step=learning_rate_steps)

        grid = list(ParameterGrid(grid))

        self.models.extend(grid)

    def print_grid_search(self):

        for model in self.models:

            print("Model: ", model["model"],
                  "Solver: ", model["solver"],
                  "Learning Rate: ", model["lr"],
                  "Epochs: ", model["epochs"],
                  "Batch: ", model["batch"],
                  "Units in LSTM Layer: ", model["lstm_units"],
                  "N. of Dense Layer: ", model["dense_layers"],
                  "Units in Dense Layer: ", model["conv_units"])

    def create_nn(self, input_shape, version, dense_layers, conv_units, lstm_units, learning_rate):

        regressor = Sequential()

        if version == 1:

            regressor.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
            regressor.add(Dropout(0.2))

            # Adding a second LSTM
            regressor.add(LSTM(units=lstm_units, return_sequences=True))
            regressor.add(Dropout(0.2))

            # Adding a third LSTM
            regressor.add(LSTM(units=lstm_units, return_sequences=True))
            regressor.add(Dropout(0.2))

            # Adding a fourth LSTM
            regressor.add(LSTM(units=lstm_units))
            regressor.add(Dropout(0.2))

        elif version == 2:

            regressor.add(LSTM(units=lstm_units, input_shape=input_shape))
            regressor.add(Dropout(0.4))

        elif version == 3:

            regressor.add(tensorflow.keras.layers.Bidirectional(LSTM(units=lstm_units, input_shape=input_shape)))
            regressor.add(Dropout(0.2))

        elif version == 4:

            regressor.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
            regressor.add(BatchNormalization())

            # Adding a second LSTM
            regressor.add(LSTM(units=lstm_units, return_sequences=True))
            regressor.add(BatchNormalization())

            # Adding a third LSTM
            regressor.add(LSTM(units=lstm_units, return_sequences=True))
            regressor.add(BatchNormalization())

            # Adding a fourth LSTM
            regressor.add(LSTM(units=lstm_units))
            regressor.add(BatchNormalization())

        # Output Layer
        for _ in range(dense_layers):

            regressor.add(Dense(units=512, activation='tanh'))

        regressor.add(Dense(units=1))

        mape = tensorflow.keras.metrics.MeanAbsolutePercentageError(name='mape')
        rmse = tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')
        mse = tensorflow.keras.metrics.MeanSquaredError(name='mse')
        mae = tensorflow.keras.metrics.MeanAbsoluteError(name='mae')
        # mre = tensorflow.keras.metrics.MeanRelativeError(name='mre')
        msle = tensorflow.keras.metrics.MeanSquaredLogarithmicError(name='msle')
        csm = tensorflow.keras.metrics.CosineSimilarity(name='csm')
        lce = tensorflow.keras.metrics.LogCoshError(name='lce')

        # Compile RNN
        regressor.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate), loss=tensorflow.keras.losses.MeanSquaredError(), metrics=[mape, rmse, mse, mae, msle, csm, lce])

        return regressor

    def grid_search(self, result_path="./grid_search_results", data='VALIDATION', anchored=False):

        if not os.path.exists(result_path):

            os.mkdir(result_path)

        results = np.zeros(7)
        histories = []
        count = 1

        start = time.time()

        for model in self.models:

            def lr_scheduler(epoch, lr):

                decay_rate = model['learning_rate_decay']
                decay_step = model['learning_rate_step']

                if epoch % decay_step == 0 and epoch:

                    return lr * pow(decay_rate, epoch // decay_step)

                return lr

            callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]

            for walk in range(self.walks['N_WALKS']):

                walk_data = self.walks['WALK_{}'.format(walk)]
                if not anchored or count == 1:
                    regressor = self.create_nn(input_shape=(walk_data['TRAIN'][0].shape[1], walk_data['TRAIN'][0].shape[2]),
                                               version=model['model'],
                                               dense_layers=model['dense_layers'],
                                               conv_units=model['conv_units'],
                                               lstm_units=model['lstm_units'],
                                               learning_rate=model['learning_rate'])

                print("\nmodel:", count, "/", len(self.models) * self.walks['N_WALKS'], "\nwalk:", walk, "\n")
                histories.append(regressor.fit(x=walk_data['TRAIN'][0],
                                               y=walk_data['TRAIN'][1],
                                               epochs=model['epochs'],
                                               batch_size=model['batch'],
                                               validation_data=walk_data['VALIDATION'],
                                               callbacks=callbacks,
                                               shuffle=False))

                walk_data['REGRESSOR'] = regressor

                count += 1

            model_path = os.path.join(result_path, 'model_{}'.format(count - 1))
            self.evaluate(data, result_path=model_path)
            json.dump(model, open(os.path.join(model_path + "_" + data, 'params.json'), 'w'))

            results = np.vstack((results, self.walks['RESULTS']['METRICS'].values))

        elapsed = time.time() - start
        hours = int(elapsed // 3600)
        mins = int((elapsed - (hours * 3600)) // 60)
        secs = int((elapsed - (hours * 3600) - (mins * 60)))

        if hours < 10: hours = "0" + str(hours)
        else:hours = str(hours)

        if mins < 10: mins = "0" + str(mins)
        else:mins = str(mins)

        if secs < 10: secs = "0" + str(secs)
        else: secs = str(secs)

        fp = open(os.path.join(result_path, 'elapsed.txt'), 'w')
        fp.write("Done in {}:{}:{}".format(hours, mins, secs))

        print()
        print("Done in {}:{}:{}".format(hours, mins, secs))
        print("All results:")
        results = pd.DataFrame(data=results[1:], columns=['MAPE', 'RMSE', 'MSE', 'MAE', 'MSLE', 'CSM', 'LCE'])
        results.to_csv(path_or_buf=os.path.join(result_path, 'all_models_overall.csv'), sep='\t')
        print(results)

    def fit(self):

        pass

    def evaluate(self, data='TEST', result_path='./evaluate_results'):

        result_path = result_path + "_" + data

        if not os.path.exists(result_path):

            os.mkdir(result_path)

        self.walks['RESULTS'] = {}
        self.walks['RESULTS']['METRICS'] = pd.DataFrame(data=np.asarray([[0, 0, 0, 0, 0, 0, 0]]), columns=['MAPE', 'RMSE', 'MSE', 'MAE', 'MSLE', 'CSM', 'LCE'])

        n_walks = self.walks['N_WALKS']
        walk_path = os.path.join(result_path, 'walk')

        print('Evaluating regressor on {} set:'.format(data))
        for walk in range(n_walks):

            walk_data = self.walks['WALK_{}'.format(walk)]
            self.evaluate_walk(walk=walk, data=data, result_path=walk_path)

            print("Walk {} results: ".format(walk))
            print(walk_data['RESULTS']['METRICS'])
            print()

            self.walks['RESULTS']['METRICS'] += walk_data['RESULTS']['METRICS'] / self.walks['N_WALKS']

        self.walks['RESULTS']['METRICS'].to_csv(path_or_buf=os.path.join(result_path, 'overall.csv'), sep='\t')

        print("Overall results: ")
        print(self.walks['RESULTS']['METRICS'])
        print()

    def evaluate_walk(self, walk, data='TEST', result_path='./'):

        result_path = result_path + "_" + str(walk)

        if not os.path.exists(result_path):

            os.mkdir(result_path)

        walk_data = self.walks['WALK_{}'.format(walk)]
        regressor = walk_data['REGRESSOR']

        pred = regressor.predict(x=walk_data[data][0])
        pred = np.reshape(pred, (pred.shape[0],))
        y = walk_data[data][1]

        walk_data['RESULTS'] = {}
        walk_data['RESULTS']['PRED'] = pred
        walk_data['RESULTS']['METRICS'] = metrics(y=y, pred=pred)

        walk_data['RESULTS']['METRICS'].to_csv(path_or_buf=os.path.join(result_path, 'walk_{}.csv'.format(walk)), sep='\t')

        self.plot_prediction_walk(walk=walk, data=data, result_path=result_path)

    def predict(self):

        pass

    def func(self):

        return 0, 0

    def plot_prediction_walk(self, walk, data='TEST', result_path='./'):

        y_denorm, pred_denorm = self.denormalize_pred_walk(walk=walk, data=data)

        start_year = self.dataset.index[0].year + walk
        train = self.split[0]
        val = self.split[1]
        test = self.split[2]

        if self.label == 'LR':

            if data == 'TRAIN':

                open = self.dataset[str(start_year):str(start_year + train - 1)]['Open'].values
                open = open[self.lookback:]

            elif data == 'VALIDATION':

                open = self.dataset[str(start_year + train):str(start_year + train + val - 1)]['Open'].values

            else:

                open = self.dataset[str(start_year + train + val):str(start_year + train + val + test - 1)]['Open'].values

            pred_close = pred_denorm * open + open

            plt.figure()
            plt.plot(open, color='red', label='Real {} Stock Prices'.format(self.stock_name))
            plt.plot(pred_close, color='blue', label='Predicted {} Stock Prices'.format(self.stock_name))
            plt.title('{} Stock Prices Prediction - walk {}'.format(self.stock_name, walk))
            plt.xlabel('Time')
            plt.ylabel('{} Stock Price'.format(self.stock_name))

            # plt.show()

            path = os.path.join(result_path, 'walk_{}_close_pred.png'.format(walk))
            plt.savefig(fname=path, dpi=100)
            plt.close()

        walk_data = self.walks['WALK_{}'.format(walk)]
        y = walk_data[data][1]
        pred = walk_data['RESULTS']['PRED']

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(y_denorm, color='red', label='Real {} Stock Prices'.format(self.stock_name))
        plt.plot(pred_denorm, color='blue', label='Predicted {} Stock Prices'.format(self.stock_name))
        plt.title('{} Stock Prices Prediction - walk {}'.format(self.stock_name, walk))
        plt.xlabel('Time')
        plt.ylabel('{} Stock Price'.format(self.stock_name))

        plt.subplot(2, 1, 2)
        plt.plot(y, color='red', label='Real {} Stock Prices'.format(self.stock_name))
        plt.plot(pred, color='blue', label='Predicted {} Stock Prices'.format(self.stock_name))
        plt.title('{} Stock Prices Prediction - walk {}'.format(self.stock_name, walk))
        plt.xlabel('Time')
        plt.ylabel('{} Stock Price'.format(self.stock_name))

        # plt.show()

        result_path = os.path.join(result_path, 'walk_{}.png'.format(walk))
        plt.savefig(fname=result_path, dpi=100)
        plt.close()

    def denormalize_pred_walk(self, walk, data='VALIDATION'):

        walk_data = self.walks['WALK_{}'.format(walk)]
        y = walk_data[data][1]
        pred = walk_data['RESULTS']['PRED']

        if self.options & stock_dataset.PRE_DIFFERENCING and self.norm_options["DIFF_PER"] == stock_dataset.DIFF_AFTER:

            y = pd.DataFrame(data=y, columns=['Val'])
            print(y['Val'].values)
            y = stock_dataset.rev_difx(y, self.norm_options["ORDER"])
            y = y['Val'].values

            pred = pd.DataFrame(data=pred, columns=['Val'])
            print(pred['Val'].values)
            pred = stock_dataset.rev_difx(pred, self.norm_options["ORDER"])
            pred = pred['Val'].values

        if self.options & stock_dataset.PRE_NORMALIZE:

            if self.norm_options["METHOD"] == stock_dataset.NORM_MIN_MAX:

                low = self.norm_options["HIGH_LOW"][0]
                high = self.norm_options["HIGH_LOW"][1]

                y = (((y - low) / (high - low)) * (walk_data['STD_PARAMS']['MAX'][0] - walk_data['STD_PARAMS']['MIN'][0]) + walk_data['STD_PARAMS']['MIN'][0])
                pred = (((pred - low) / (high - low)) * (walk_data['STD_PARAMS']['MAX'][0] - walk_data['STD_PARAMS']['MIN'][0]) + walk_data['STD_PARAMS']['MIN'][0])

            elif self.norm_options["METHOD"] == stock_dataset.NORM_Z_SCORE:

                y = (y * walk_data["STD_PARAMS"]["STD"][0]) + walk_data["STD_PARAMS"]["MEAN"][0]
                pred = (pred * walk_data["STD_PARAMS"]["STD"][0]) + walk_data["STD_PARAMS"]["MEAN"][0]

        if self.options & stock_dataset.PRE_DIFFERENCING and self.norm_options["DIFF_PER"] == stock_dataset.DIFF_BEFORE:

            y = pd.DataFrame(data=y, columns=['Val'])
            print(y['Val'].values)
            y = stock_dataset.rev_difx(y, self.norm_options["ORDER"])
            y = y['Val'].values

            pred = pd.DataFrame(data=pred, columns=['Val'])
            print(pred['Val'].values)
            pred = stock_dataset.rev_difx(pred, self.norm_options["ORDER"])
            pred = pred['Val'].values

        return y, pred

    def denormalize_data_walk(self, walk, data='VALIDATION'):

        walk_data = self.walks['WALK_{}'.format(walk)]
        x = walk_data[data][0]

        if self.options & stock_dataset.PRE_DIFFERENCING and self.norm_options["DIFF_PER"] == stock_dataset.DIFF_AFTER:

            x = pd.DataFrame(data=x, columns=['Val'])
            print(x['Val'].values)
            x = stock_dataset.rev_difx(x, self.norm_options["ORDER"])
            x = x['Val'].values

        if self.options & stock_dataset.PRE_NORMALIZE:

            if self.norm_options["METHOD"] == stock_dataset.NORM_MIN_MAX:

                low = self.norm_options["HIGH_LOW"][0]
                high = self.norm_options["HIGH_LOW"][1]

                x = (((x - low) / (high - low)) * (walk_data['STD_PARAMS']['MAX'][0] - walk_data['STD_PARAMS']['MIN'][0]) + walk_data['STD_PARAMS']['MIN'][0])
            elif self.norm_options["METHOD"] == stock_dataset.NORM_Z_SCORE:

                x = (x * walk_data["STD_PARAMS"]["STD"][0]) + walk_data["STD_PARAMS"]["MEAN"][0]

        if self.options & stock_dataset.PRE_DIFFERENCING and self.norm_options["DIFF_PER"] == stock_dataset.DIFF_BEFORE:

            x = pd.DataFrame(data=x, columns=['Val'])
            print(x['Val'].values)
            x = stock_dataset.rev_difx(x, self.norm_options["ORDER"])
            x = x['Val'].values

        return x


def metrics(y, pred):

    mape = tensorflow.keras.metrics.MeanAbsolutePercentageError()
    rmse = tensorflow.keras.metrics.RootMeanSquaredError()
    mse = tensorflow.keras.metrics.MeanSquaredError()
    mae = tensorflow.keras.metrics.MeanAbsoluteError()
    msle = tensorflow.keras.metrics.MeanSquaredLogarithmicError()
    csm = tensorflow.keras.metrics.CosineSimilarity()
    lce = tensorflow.keras.metrics.LogCoshError()

    mape.update_state(y, pred)
    rmse.update_state(y, pred)
    mse.update_state(y, pred)
    mae.update_state(y, pred)
    msle.update_state(y, pred)
    csm.update_state(y, pred)
    lce.update_state(y, pred)

    # print("MAPE:", mape.result().numpy())
    # print("RMSE:", rmse.result().numpy())
    # print("MSE:", mse.result().numpy())
    # print("MAE:", mae.result().numpy())
    # print("MSLE:", msle.result().numpy())
    # print("CSM:", csm.result().numpy())
    # print("LCE:", lce.result().numpy())

    columns = ['MAPE', 'RMSE', 'MSE', 'MAE', 'MSLE', 'CSM', 'LCE']
    data = np.asarray([[mape.result().numpy(), rmse.result().numpy(), mse.result().numpy(), mae.result().numpy(), msle.result().numpy(), csm.result().numpy(), lce.result().numpy()]])

    return pd.DataFrame(data=data, columns=columns)


def guessing_test(pred, label, offset=0):

    choice = np.sum((pred > offset))
    choice += np.sum((pred < -offset))
    increment = np.sum((pred > offset) & (label > 0))
    decrement = np.sum((pred < -offset) & (label < 0))

    print("number of right increment:", increment)
    print("number of right decrement:", decrement)
    print("Number of elements:", len(label))

    print("Accuracy:", round((increment+decrement)/len(label), 2))
    print("Accuracy on choice:", round((increment+decrement)/choice, 2))
