# Let's predict the true near term price of a corporate monster without bullshit.

It's fun to short big companies :_)

An implementation of Convolutional Neural Network (CNN) and Tensor Neural Network (TNN) to classify and predict financial data. The scripts process input data and use Keras with a TensorFlow backend to construct, train, and evaluate deep learning models.

## Dependencies

- Python 3.x
- pandas
- numpy
- Keras
- TensorFlow

Ensure you have the above Python packages installed using the following:

## Files

### cnn.py

This script contains the implementation of a Convolutional Neural Network with the Keras library.

Functions

make_model(input_size): Defines and returns the CNN model structure.
data_generator(data, size): A generator function for feeding data to the Keras model during training and evaluation.
train(input_size): Compiles and trains the CNN model.
predict(input_size, weight): Loads the trained weights and evaluates the model on the test dataset.
prepare_input(company): Calls functions from preprocess_training_data to prepare the input data for training and testing.

### tnn.py

This script contains the implementation of a Tensor Neural Network with Keras.

Classes

Tnet(Layer): A custom Keras layer that implements the Tnet layer of a tensor neural network.
Functions

get_regularizer(W): Returns a Keras regularizer for a given tensor W.
make_model(train): Defines and returns the TNN model structure.

``` bash
python cnn.py prepare_input <company_name>
python cnn.py train
python cnn.py predict <epoch_number>
```

Ensure the input data is correctly placed in the expected directories, and the necessary preprocessing scripts are correctly imported.
The CNN uses embedding layers and convolutional layers to process input sequences representing financial events over different time horizons.

### Author

Thomas F McGeehan V (TFMV)
