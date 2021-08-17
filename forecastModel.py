import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import time



class forecastModel:

    def __init__(self, dataset, years_split, lookback=60,scaler=None):
        self.dataset = dataset
        self.years_split = years_split
        self.train_years = years_split[0]
        self.validation_years = years_split[1]
        self.test_years = years_split[2]
        self.modeles_parameters = []
        self.scaler = scaler
        self.lookback = lookback

        self.train_sets, self.validation_sets, self.test_sets, self.n_walk = dataset.walk_forward(self.years_split)

        # Split set in sequence
        X_train = []
        y_train = []
        X_validation = []
        y_validation = []
        X_test = []
        y_test = []

        val_pred = 0  # firts column in pandas
        for i in range(self.n_walk):
            train_seq = self._get_sequences(self.train_sets[i], self.lookback)
            X_train.append(train_seq[:-1])
            y_train.append(train_seq[1:, -1, val_pred])  # get label from first element of next sequence

            validation_seq = self._get_sequences(self.validation_sets[i], self.lookback, padding=train_seq[-1])
            X_validation.append(validation_seq[:-1])
            y_validation.append(validation_seq[1:, -1, val_pred])

            test_seq = self._get_sequences(self.test_sets[i], lookback, padding=validation_seq[-1])
            X_test.append(test_seq[:-1])
            y_test.append(test_seq[1:, -1, val_pred])

        self.X_train, self.y_train = np.asarray(X_train), np.asarray(y_train)
        self.X_validation, self.y_validation = np.asarray(X_validation), np.asarray(y_validation)
        self.X_test, self.y_test = np.asarray(X_test), np.asarray(y_test)


    def _get_sequences(self, array, window, padding=[]):
        if (len(padding) > 0):
            # padding = np.asarray(padding)
            array = np.vstack((padding, array))
        shape = (array.shape[0] - window + 1, window, array.shape[1])
        strides = array.strides[:-1] + array.strides

        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)



    def add_grid_search(self, models=[1], solvers=['adam'], lrs=[0.001], epochs=[25], batches=[32], lstm_units=[50],
                        dense_layers=[0], conv_units=[0]):
        grid = dict(model=models,
                    solver=solvers,
                    lr=lrs,
                    epochs=epochs,
                    batch=batches,
                    lstm_units=lstm_units,
                    dense_layers=dense_layers,
                    conv_units=conv_units)

        grid = list(ParameterGrid(grid))

        self.models.append(grid)

    def print_grid_search(self):
        for model in self.models:
            self.print_model(model)

    def print_model(self,model):
        print("Model:", model["model"], "Solver:", model["solver"], "Learning Rate:", model["lr"],
              "Epochs:", model["epochs"], "Batch:", model["batch"], "Units in LSTM Layer:", model["lstm_units"],
              "N. of Dense Layer:", model["dense_layers"], "Units in Dense Layer:", model["conv_units"])




    def get_best_model(self):

        def lr_scheduler(epoch, lr):
            decay_rate = 0.85
            decay_step = 5

            if epoch % decay_step == 0 and epoch:
                return lr * pow(decay_rate, np.floor(epoch / decay_step))

            return lr

        callbacks = [LearningRateScheduler(lr_scheduler, verbose=1),
                     EarlyStopping(monitor='val_loss', patience=5, verbose=1)]

        result = []
        history_collection = []
        min_rmse = 99999

        print("Number of models:", len(self.models))
        tot_time = time.time()
        history = []
        count = 0
        # train all models
        for model in self.models:
            print("\n***************************\n", model)
            set_regressors = []

            model_time = time.time()
            rmse_tot = 0
            mape_tot = 0
            # walk forward
            for i in range(self.n_walk - 1):
                regressor = self.create_nn(input_shape=(self.X_train[0].shape[1], self.X_train[0].shape[2]),
                                           version=model['model'],
                                           dense_layers=model['dense_layers'],
                                           conv_units=model['conv_units'],
                                           lstm_units=model['lstm_units'],
                                           learning_rate=model['lr'])
                print("\nmodel:", count, "/", len(self.models), "\nwalk:", i, "\n")
                history.append(
                    regressor.fit(self.y_train[i], self.y_train[i], epochs=model['epochs'], batch_size=model['batch'],
                                  validation_data=(self.X_validation[i], self.y_validation[i]), callbacks=callbacks))

                # Prediction on test set
                prediction = regressor.predict(self.X_test[i])
                y = self.y_test[i].reshape(-1, 1)
                if self.scaler is not None:
                    prediction = self.scaler.inverse_transform(np.hstack((prediction,
                                                                          np.zeros((len(self.X_test[i]), self.X_test[0].shape[2] - 1)))))
                    y = self.scaler.inverse_transform(np.hstack((y, np.zeros((len(y), self.X_test[0].shape[2] - 1)))))

                # Evaluating our model
                # plot_prediction(name, y[:, 0], prediction[:, 0])
                rmse, mape = self.return_rmse(y[:, 0], prediction[:, 0])
                rmse_tot += rmse
                mape_tot += mape
                print("\nTOT:")
                print(model)
                print("RMSE: {}.".format(rmse))
                print("MAPE: {}.".format(mape))
                result.append([(rmse, mape), model, int((time.time() - model_time) / 60)])
                print("The model has been generated in", int((time.time() - model_time) / 60), "min")
                set_regressors.append(regressor)

            if rmse_tot < min_rmse:
                min_rmse = rmse_tot
                best_set_regressor = set_regressors
                print("******** THIS IS THE BEST MODEL*********")
                print(model)
                print("****************************************")

            print("\nMean:")
            print(model)
            print("RMSE mean: {}.".format(rmse))
            print("MAPE mean: {}.".format(mape))

        print("All models has been generated in", int((time.time() - tot_time) / 60), "min")

        return result, history_collection, best_set_regressor




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

            regressor.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
            regressor.add(Dropout(0.2))
            regressor.add(Conv1D(filters=conv_units, kernel_size=3))
            regressor.add(BatchNormalization())
            regressor.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))

            # Adding a second LSTM
            regressor.add(LSTM(units=lstm_units, return_sequences=True))
            regressor.add(Dropout(0.2))
            regressor.add(Conv1D(filters=conv_units, kernel_size=3))
            regressor.add(BatchNormalization())
            regressor.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))

            # Adding a third LSTM
            regressor.add(LSTM(units=lstm_units, return_sequences=True))
            regressor.add(Dropout(0.2))

            # Adding a fourth LSTM
            regressor.add(LSTM(units=lstm_units))
            regressor.add(Dropout(0.2))

        elif version == 3:

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

            regressor.add(Dense(units=128, activation='tanh'))

        regressor.add(Dense(units=1))

        # Compile RNN
        regressor.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae', 'mape'])

        return regressor



    # Results
    def plot_prediction(self, name, y_test, pred):

      plt.plot(y_test, color = 'red', label = 'Real {} Stock Price'.format(name))
      plt.plot(pred, color = 'blue', label = 'Predicted {} Stock Price'.format(name))
      plt.title('{} Stock Price Prediction'.format(name))
      plt.xlabel('Time')
      plt.ylabel('{} Stock Price'.format(name))
      plt.show()


    def metrics(self,test, predicted):
        rmse = math.sqrt(mean_squared_error(test, predicted))
        mape = np.mean(abs((test - predicted) / test))

        print("RMSE: {}.".format(rmse))
        print("MAPE: {}.".format(mape))

        return rmse, mape



    def plot_prediction_Close(self, real, regressor, X_test_set, y_test_set, n_walk, sc=None):
        #    for i in range(n_walk):
        real = real[-252:]
        real_one_day_before = np.roll(real, 1)[1:]
        i = 7
        prediction = regressor.predict(X_test_set[i])

        if sc is not None:
            prediction = sc.inverse_transform(np.hstack((prediction,
                                                         np.zeros((len(X_test_set[i]), X_test_set[0].shape[2] - 1)))))

            y = y_test_set[i].reshape(-1, 1)
            y = sc.inverse_transform(np.hstack((y, np.zeros((len(y), X_test_set[0].shape[2] - 1)))))
            self.plot_prediction("name", real[1:], prediction[:, 0] * real_one_day_before + real_one_day_before)
            self.plot_prediction("name", y[:, 0], prediction[:, 0])
        else:
            self.plot_prediction("name", real[1:], prediction * real_one_day_before + real_one_day_before)
            for j in range(len(prediction[:10])):
                print(prediction[j], prediction[j] * real_one_day_before[j] + real_one_day_before[j], real[j + 1],
                      y_test_set[i])

        return prediction, y
