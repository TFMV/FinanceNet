# cnn.py
import datetime
import numpy as np
import yfinance as yf
import random
import ast
import sys
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Convolution1D, MaxPooling1D, Input, Flatten, Activation
from keras.layers import BatchNormalization, Dropout, ELU
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

import preprocess_training_data as pp

def get_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock['rets'] = stock['Adj Close'].pct_change()
    return stock

def make_model(input_size):
    D = input_size  # Length of observations
    N1, N2 = 30, 7  # context lengths
    elu = ELU(alpha=0.8)

    # Long Term Events
    longT_ = Input(shape=(N1, D))
    longT = Convolution1D(64, 3, padding='valid', activation=elu)(longT_)
    longT = MaxPooling1D()(longT)
    longT = Flatten()(longT)

    # Mid Term Events
    midT_ = Input(shape=(N2, D))
    midT = Convolution1D(64, 3, padding='valid', activation=elu)(midT_)
    midT = MaxPooling1D()(midT)
    midT = Flatten()(midT)

    # Short Term Events
    shortT_ = Input(shape=(D,))
    prev = Input(shape=(2,))

    # Combine feature vectors
    d = K.concatenate([longT, midT, shortT_, prev], axis=1)

    # Feedforward
    hidden = Dense(100, activation=elu)(d)
    hidden = Dropout(0.25)(hidden)
    output = Dense(2, activation='softmax')(hidden)

    # Define model
    model = Model(inputs=[longT_, midT_, shortT_, prev], outputs=output)
    return model

def data_generator(data, size):
    """Generate input data."""
    df = pd.read_csv(f'./data/cnn-{data}.csv')
    df.set_index(['Date'], inplace=True)
    samples = len(df)
    i, j = -1, 30
    while True:
        if data == 'test':
            i += 1
            j += 1
        else:
            i = random.randrange(samples - 31)
            j = i + 31

        longTerm = df[i:j]
        ltX_, y_prev, y = longTerm.Embedding.values[:30], longTerm.Movement.values[-3:-1], longTerm.Movement.values[-1]
        ltX = np.array(list(map(ast.literal_eval, ltX_)))
        mtX = ltX[-7:]
        stX = ltX[-1]

        a, b, c, prev = np.reshape(ltX, [1, 30, size]), np.reshape(mtX, [1, 7, size]), np.reshape(stX, [1, size]), np.reshape(y_prev, [1, 2])
        y_ = np_utils.to_categorical([int(y)], 2)
        yield ([a, b, c, prev], y_)

def train(input_size, epochs=50, batch_size=32):
    model = make_model(input_size)
    optimizer = Adam(lr=0.0001, decay=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    class LossHistory(Callback):
        def on_train_begin(self, logs=None):
            self.matthews_correlation = []

        def on_epoch_end(self, epoch, logs=None):
            self.matthews_correlation.append(logs.get('matthews_correlation'))

    history = LossHistory()
    checkpoint = ModelCheckpoint(filepath='./cnn-weights/checkpoint-{epoch:02d}.hdf5', save_weights_only=True, save_best_only=True)

    model.fit_generator(data_generator(data='training', size=input_size), 
                        validation_data=data_generator(data='validation', size=input_size),
                        steps_per_epoch=2031 // batch_size, 
                        validation_steps=216 // batch_size,
                        epochs=epochs,
                        callbacks=[history, checkpoint],
                        verbose=1)

    model.save_weights('./cnn-weights/cnn-weights.h5')

def predict(input_size, weight):
    model = make_model(input_size)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(f'./cnn-weights/checkpoint-{weight}.hdf5')
    preds = model.evaluate_generator(data_generator(data='test', size=input_size), steps=248 // 32)
    for i, metric in enumerate(model.metrics_names):
        print(f"{metric}: {preds[i]}")

def prepare_input(ticker):
    pp.process_sandp(ticker)
    df = pp.add_dates(ticker)
    pp.construct_output(df, copy_paper=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python cnn.py <train/predict> <ticker> [weight]")
        sys.exit(1)
    
    mode = sys.argv[1]
    ticker = sys.argv[2]
    prepare_input(ticker)

    if mode == 'train':
        train(input_size=5)
    elif mode == 'predict':
        if len(sys.argv) < 4:
            print("Usage: python cnn.py predict <ticker> <weight>")
            sys.exit(1)
        weight = sys.argv[3]
        predict(input_size=5, weight=weight)
