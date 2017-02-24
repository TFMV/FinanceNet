import pandas as pd
import numpy as np
import random
import ast
import preprocess_training_data as pp
 
from keras import backend as BE
from keras.models import Model
from keras.layers import  Dense, Convolution1D, MaxPooling1D, Input, merge, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU
from keras.regularizers import l1l2
 
import sys
 
def make_model(input_size):
    D = input_size  # Length of observations
    N1, N2 = 30, 7 # context lengths
    elu = ELU(alpha=0.8)
 
    # Longs Term Events
    longT_ = Input(shape=(N1, D))
 
    longT = Convolution1D(64, 3, border_mode='valid', input_shape=(N1, D), activation=elu)(longT_)
    longT = MaxPooling1D()(longT)
    longT = Flatten()(longT)
 
    # Mid Term Events
    midT_ = Input(shape=(N2, D))
    midT = Convolution1D(64, 3, border_mode='valid', input_shape=(N2, D), activation=elu)(midT_)
    midT = MaxPooling1D()(midT)
    midT = Flatten()(midT)
 
    # Short Term Events
    shortT_ = Input(shape=(D, ))
    # c = BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.99, beta_init='one', gamma_init='one', gamma_regularizer='l2', beta_regularizer='l2')(c_)
 
    # Previous movement
    prev = Input(shape=(2,))
 
    # # Combine feature vectors
    # d = merge([longT,midT,shortT_], mode='concat', concat_axis=1)
    d = merge([longT,midT,shortT_,prev], mode='concat', concat_axis=1) # With previous movemnt
 
    # Feedfoward
    hidden = Dense(100, activation=elu)(d)
    # hidden = Dense(100, activation='sigmoid')(d)
    # hidden = Dropout(0.5)(hidden)
    # hidden = Dense(50, activation=elu)(hidden)
    hidden = Dropout(0.25)(hidden)
    # hidden = Dense(10, activation=elu)(hidden)
 
    # hidden = BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.99, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer='l1')(hidden)
    output = Dense(2, activation='softmax')(hidden)
    # output = Dense(2, activation='sigmoid')(hidden)
 
    # Define model
    model = Model(input=[longT_, midT_, shortT_, prev], output=output)
    return model
 
 
# Generator for train/dev/test sets
def data_generator(data, size):
    """
    Generate input data.
    Skip ahead past weekends when
    there is no S&P change.
 
    data can be 'training', 'valdiation', or 'test'
    """
    # Read in input data
    df = pd.read_csv('./data/cnn-{}.csv'.format(data))
    df.set_index(['Date'], inplace=True)
    samples = len(df)
    i, j = -1, 30
    while 1:
        if data == 'test':
            i+=1
            j+=1
 
        else:
            i = random.randrange(samples-31)
            j = i+31
 
        # Long term events
        longTerm = df[i:j]
        ltX_, y_prev, y = longTerm.Embedding.values[:30], longTerm.Movement.values[-3:-1],longTerm.Movement.values[-1]
        ltX = np.array(list(map(ast.literal_eval, ltX_)))
 
        # Mid term events
        mtX = ltX[-7:]
 
        # Short term events
        stX = ltX[-1]
 
        # Generate data
        # if not np.isnan(y):
        a, b, c, prev = np.reshape(ltX, [1, 30, size]), np.reshape(mtX, [1, 7, size]), np.reshape(stX, [1, size]), np.reshape(y_prev,[1,2])
        y_ = np_utils.to_categorical([int(y)], 2)
        yield ([a, b, c, prev], y_)
 
 
def train(input_size):
    # Compile model
    model = make_model(input_size=input_size)
    # model.summary()
 
    # optimizer = SGD(lr=0.001)
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'matthews_correlation'])
 
    # Alter LossHistory class to show MCC after each epoch
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.matthews_correlation = []
        def on_epoch_end(self, batch, logs={}):
            self.matthews_correlation.append(logs.get('matthews_correlation'))
    history = LossHistory()
 
    checkpoint = ModelCheckpoint(filepath='./cnn-weights/checkpoint-{epoch:02d}.hdf5',\
                                                    save_weights_only=True, save_best_only=True)
 
    # Train
    model.fit_generator(data_generator(data='training', size=input_size), \
        validation_data=data_generator(data='validation', size=input_size),\
        samples_per_epoch=2031, nb_val_samples=216, nb_epoch=50,      \
        callbacks=[history, checkpoint], verbose=1, max_q_size=32)
 
    model.save_weights('./cnn-weights/cnn-weights.h5')
 
 
def predict(input_size, weight):
    model = make_model(input_size=input_size)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', 'matthews_correlation'])
 
    model.load_weights('./cnn-weights/checkpoint-{}.hdf5'.format(weight))
 
    preds = model.evaluate_generator(data_generator(data='test', size=input_size), val_samples=248, max_q_size=4)
 
    for i in range(len(preds)):
        print("{}: {}".format(model.metrics_names[i], preds[i]))
    # print(model.predict_generator(data_generator(data='test', size=input_size), val_samples=278))
 
def prepare_input(company):
    pp.process_sandp(company)
    df = pp.add_dates(company)
    pp.consctruct_output(df, copy_paper=True)
 
if __name__ == '__main__':
    prepare_input(sys.argv[2])
    if sys.argv[1] == 'train':
        train(input_size=5)
    elif sys.argv[1] == 'predict':
        predict(input_size=5, weight=sys.argv[2])
