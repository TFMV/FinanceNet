import sys
import datetime
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Input, Flatten, Dropout, ELU, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def get_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock['rets'] = stock['Adj Close'].pct_change()
    stock.loc[:, 'rets'] = stock['rets'].fillna(0)  # Handle NaN values
    stock['Movement'] = np.where(stock['rets'] > 0, 1, 0)  # Example movement calculation
    return stock

def make_model(input_size):
    N1, N2 = 30, 7  # context lengths
    elu = ELU(alpha=0.8)

    # Long Term Events
    longT_ = Input(shape=(N1, input_size))
    longT = Conv1D(64, 3, padding='valid', activation=elu)(longT_)
    longT = MaxPooling1D()(longT)
    longT = Flatten()(longT)

    # Mid Term Events
    midT_ = Input(shape=(N2, input_size))
    midT = Conv1D(64, 3, padding='valid', activation=elu)(midT_)
    midT = MaxPooling1D()(midT)
    midT = Flatten()(midT)

    # Short Term Events
    shortT_ = Input(shape=(input_size, ))

    # Previous movement
    prev = Input(shape=(2, ))

    # Combine feature vectors
    d = concatenate([longT, midT, shortT_, prev])

    # Feedforward
    hidden = Dense(100, activation=elu)(d)
    hidden = Dropout(0.25)(hidden)
    output = Dense(2, activation='softmax')(hidden)

    # Define model
    model = Model(inputs=[longT_, midT_, shortT_, prev], outputs=output)
    return model

def data_generator(stock_data, size):
    """Generate input data."""
    samples = len(stock_data)
    i, j = -1, 30
    while True:
        if i + 31 >= samples:
            i, j = -1, 30  # Reset for another epoch
        i += 1
        j = i + 31

        # Long term events
        longTerm = stock_data.iloc[i:j]
        if len(longTerm) < 31:
            continue

        ltX_, y_prev, y = longTerm['rets'].values[:30], longTerm['Movement'].values[-3:-1], longTerm['Movement'].values[-1]
        ltX = np.array(list(ltX_)).reshape(-1, 1)

        # Mid term events
        mtX = ltX[-7:]

        # Short term events
        stX = np.array([ltX[-1]]).reshape(1, 1)  # Ensure stX is a 2D array

        print(f"ltX shape: {ltX.shape}, mtX shape: {mtX.shape}, stX shape: {stX.shape if hasattr(stX, 'shape') else (size,)}")

        # Ensure shapes match
        if ltX.shape[0] == 30 and mtX.shape[0] == 7:
            a, b, c, prev = np.reshape(ltX, [1, 30, 1]), np.reshape(mtX, [1, 7, 1]), np.reshape(stX, [1, 1]), np.reshape(y_prev, [1, 2])
            y_ = to_categorical([int(y)], 2)
            yield ([a, b, c, prev], y_)

def create_dataset(stock_data, input_size):
    """Create TensorFlow dataset from generator."""
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 30, input_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 7, input_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None, input_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(lambda: data_generator(stock_data, input_size), output_signature=output_signature)

def train(input_size, ticker):
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365*10)).strftime('%Y-%m-%d')  # 10 years ago
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Compile model
    model = make_model(input_size=input_size)
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='./cnn-weights/checkpoint-{epoch:02d}.weights.h5', save_weights_only=True, save_best_only=True)

    # Create datasets
    train_dataset = create_dataset(stock_data, input_size).batch(32)
    val_dataset = create_dataset(stock_data, input_size).batch(32)

    # Train
    model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[checkpoint], verbose=1)
    model.save_weights('./cnn-weights/cnn-weights.h5')

def predict(input_size, ticker, weight):
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365*10)).strftime('%Y-%m-%d')  # 10 years ago
    stock_data = get_stock_data(ticker, start_date, end_date)

    model = make_model(input_size=input_size)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(f'./cnn-weights/{weight}')

    test_dataset = create_dataset(stock_data, input_size).batch(32)

    preds = model.evaluate(test_dataset)

    for i in range(len(preds)):
        print(f"{model.metrics_names[i]}: {preds[i]}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python cnn.py <train/predict> <ticker> [weight]")
        sys.exit(1)

    mode = sys.argv[1]
    ticker = sys.argv[2]

    if mode == 'train':
        train(input_size=1, ticker=ticker)  # Using 1 feature (returns)
    elif mode == 'predict':
        if len(sys.argv) < 4:
            print("Usage: python cnn.py predict <ticker> <weight>")
            sys.exit(1)
        weight = sys.argv[3]
        predict(input_size=1, ticker=ticker, weight=weight)
