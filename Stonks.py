#Alpaca stuff

# import alpaca_trade_api as tradeapi
# import pandas as pd
# import pandas_ta as ta
# APCA_API_BASE_URL='https://data.alpaca.markets/v1'
# APCA_API_KEY_ID='PK6TO93TGBBXNJZIXLFL'
# APCA_Client_ID='9538afc06996eada38f6401b0685c27b'
# APCA_API_SECRET_KEY='ryb0UXMkRsZwEhGgsDLE9ZwQdG2gSP8I9JE3LEpM'
# APCA_CLIENT_SECRET_KEY='784a4c3c4b1e053efbd34f0b7ec8d78d8774d7ad'
# api = tradeapi.REST(key_id=APCA_API_KEY_ID,
#         secret_key=APCA_API_SECRET_KEY,
#         base_url=APCA_API_BASE_URL, api_version='v2',)
#
# # Get daily price data for AAPL over the last 5 trading days.
# # barset = api.get_barset('QQQ', 'day', limit=5)
# # aapl_bars = barset['QQQ']
# #
# # # See how much AAPL moved in that timeframe.
# # week_open = aapl_bars[0].o
# # week_close = aapl_bars[-1].c
# # percent_change = (week_close - week_open) / week_open * 100
# # print('AAPL moved {}% over the last 5 days'.format(percent_change))
#
# stockdf = api.get_barset(
#         'AAPL',
#         'day',
#         #start=pd.Timestamp('2020-01-01 09:00',tz='America/New_York').isoformat(),
#         until=pd.Timestamp('2020-12-31 17:00',tz='America/New_York').isoformat(),
#         limit=1000
#         )
#
# print(stockdf.df)

#Other Stuff

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib
import numpy as np
import tensorflow
endYear='2020'


AMZN = yf.download('AMZN',
                      start='2013-01-01',
                      end='2021-12-31',
                      progress=False)
# AMZN = yf.download('AMZN') for all
all_data = AMZN[['Adj Close','Open', 'High', 'Low', 'Close', 'Volume']].round(2)
all_data.head(10)
print("There are "+ str(all_data[:'2019'].shape[0]) + " observations in the training data")
print("There are "+ str(all_data[f'{endYear}':].shape[0]) + " observations in the test data")

print(all_data)
all_data['Adj Close'].plot()
# There are 1,510 and 251 observations in the training and test data respectively.


# def ts_train_test(all_data, time_steps, for_periods):
#     '''
#     input:
#       data: dataframe with dates and price data
#     output:
#       X_train, y_train: data from 2013/1/1-2018/12/31
#       X_test:  data from 2019 -
#     time_steps: # of the input time steps
#     for_periods: # of the output time steps
#     '''
#     # create training and test set
#     ts_train = all_data[:'2018'].iloc[:, 0:1].values
#     ts_test = all_data['2019':].iloc[:, 0:1].values
#     ts_train_len = len(ts_train)
#     ts_test_len = len(ts_test)
#
#     # create training data of s samples and t time steps
#     X_train = []
#     y_train = []
#     y_train_stacked = []
#     for i in range(time_steps, ts_train_len - 1):
#         X_train.append(ts_train[i - time_steps:i, 0])
#         y_train.append(ts_train[i:i + for_periods, 0])
#     X_train, y_train = np.array(X_train), np.array(y_train)
#
#     # Reshaping X_train for efficient modelling
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#
#     # Preparing to create X_test
#     inputs = pd.concat((all_data["Adj Close"][:'2018'], all_data["Adj Close"]['2019':]), axis=0).values
#     inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
#     inputs = inputs.reshape(-1, 1)
#
#     X_test = []
#     for i in range(time_steps, ts_test_len + time_steps - for_periods):
#         X_test.append(inputs[i - time_steps:i, 0])
#
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
#     return X_train, y_train, X_test
#
#
# X_train, y_train, X_test = ts_train_test(all_data, 5, 2)
# X_train.shape[0], X_train.shape[1]
#
#
# # Convert the 3-D shape of X_train to a data frame so we can see:
# X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0],X_train.shape[1])))
# y_train_see = pd.DataFrame(y_train)
# pd.concat([X_train_see,y_train_see],axis=1)
#
# # Convert the 3-D shape of X_test to a data frame so we can see:
# X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0],X_test.shape[1])))
# pd.DataFrame(X_test_see)
#
# print("There are " + str(X_train.shape[0]) + " samples in the training data")
# print("There are " + str(X_test.shape[0]) + " samples in the test data")
def ts_train_test_normalize(all_data, time_steps, for_periods):
    '''
    input:
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2013/1/1-2018/12/31
      X_test:  data from 2019 -
      sc:      insantiated MinMaxScaler object fit to the training data
    '''
    # create training and test set
    ts_train = all_data[:'2019'].iloc[:, 0:1].values
    ts_test = all_data[f'{endYear}':].iloc[:, 0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = pd.concat((all_data["Adj Close"][:'2019'], all_data["Adj Close"][f'{endYear}':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc





def simple_rnn_model(X_train, y_train, X_test, sc):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN

    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    # my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    # my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(2))  # The time step of the output

    my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    my_rnn_model.fit(X_train, y_train, epochs=100, batch_size=150, verbose=0)

    # Finalizing predictions
    rnn_predictions = my_rnn_model.predict(X_test)
    from sklearn.preprocessing import MinMaxScaler
    rnn_predictions = sc.inverse_transform(rnn_predictions)

    return my_rnn_model, rnn_predictions




def actual_pred_plot(preds):
    '''
    Plot
    the
    actual
    vs.prediction
    '''
    actual_pred = pd.DataFrame(columns=['Adj. Close', 'prediction'])
    actual_pred['Adj. Close'] = all_data.loc[f'{endYear}':, 'Adj Close'][0:len(preds)]
    actual_pred['prediction'] = preds[:, 0]

    from keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Adj. Close']), np.array(actual_pred['prediction']))

    return (m.result().numpy(), actual_pred.plot())


#actual_pred_plot(rnn_predictions)

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 5, 2)
my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)
rnn_predictions_2[1:10]
print(rnn_predictions_2)
actual_pred_plot(rnn_predictions_2)











def LSTM_model(X_train, y_train, X_test, sc):
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU, LSTM
    from keras.optimizers import SGD

    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    # my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    # my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dense(units=2))

    # Compiling
    my_LSTM_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
    # Fitting to the training set
    my_LSTM_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    LSTM_prediction = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)

    return my_LSTM_model, LSTM_prediction


my_LSTM_model, LSTM_prediction = LSTM_model(X_train, y_train, X_test, sc)
LSTM_prediction[1:10]
actual_pred_plot(LSTM_prediction)


def GRU_model(X_train, y_train, X_test, sc):
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU
    from keras.optimizers import SGD

    # The GRU architecture
    my_GRU_model = Sequential()
    # First GRU layer with Dropout regularisation
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    # my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    # my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dense(units=2))

    # Compiling the RNN
    my_GRU_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
    # Fitting to the training set
    my_GRU_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    GRU_prediction = my_GRU_model.predict(X_test)
    GRU_prediction = sc.inverse_transform(GRU_prediction)

    return my_GRU_model, GRU_prediction


my_GRU_model, GRU_prediction = GRU_model(X_train, y_train, X_test, sc)
GRU_prediction[1:10]
actual_pred_plot(GRU_prediction)


def GRU_model_regularization(X_train, y_train, X_test, sc):
    '''
    create GRU model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU
    from keras.optimizers import SGD
    from keras.layers import Dropout

    # The GRU architecture
    my_GRU_model = Sequential()
    # First GRU layer with Dropout regularisation
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Second GRU layer
    my_GRU_model.add(GRU(units=50, return_sequences=True, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))

    # Third GRU layer
    my_GRU_model.add(GRU(units=50, return_sequences=True, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Fourth GRU layer
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # The output layer
    my_GRU_model.add(Dense(units=2))
    # Compiling the RNN
    my_GRU_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
    # Fitting to the training set
    my_GRU_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    GRU_predictions = my_GRU_model.predict(X_test)
    GRU_predictions = sc.inverse_transform(GRU_predictions)

    return my_GRU_model, GRU_predictions


my_GRU_model, GRU_predictions = GRU_model_regularization(X_train, y_train, X_test, sc)
GRU_predictions[1:10]
actual_pred_plot(GRU_prediction)

