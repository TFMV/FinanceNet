# FinanceNet

FinanceNet is a comprehensive suite of tools for financial data analysis using advanced neural networks. The project includes implementations of a Convolutional Neural Network (CNN) and a Tensor Network (TNet) for processing and analyzing financial data.

![FinanceNet](assets/FinanceNet.webp)

## Algorithms

### Convolutional Neural Network (CNN)

The CNN implementation in `cnn.py` is designed to analyze financial time series data. The architecture is structured to capture long-term, mid-term, and short-term events to predict future movements based on historical data.

**Key Components:**

- **Long Term Events:**
  - Utilizes a Convolutional layer followed by MaxPooling and Flatten layers.
  - Captures long-term dependencies in the data over 30 time steps.

- **Mid Term Events:**
  - Similar to long-term, but over 7 time steps.
  - Focuses on capturing mid-term patterns.

- **Short Term Events:**
  - Direct input without convolution.
  - Captures immediate short-term changes.

- **Previous Movement:**
  - Includes previous movement data to enhance predictive power.

- **Feature Combination:**
  - Combines the outputs from long-term, mid-term, short-term, and previous movement inputs using concatenation.

- **Feedforward Neural Network:**
  - Dense layers with ELU activation and Dropout for regularization.
  - Final output layer uses softmax activation to predict the probability of upward or downward movement.

**Training Process:**

- Utilizes Adam optimizer and categorical_crossentropy loss function.
- Implements a custom data generator to yield batches of training data.
- Incorporates callbacks such as ModelCheckpoint to save the best model weights during training.

### Tensor Network (TNet)

The TNet implementation in `tnet.py` is a custom neural network layer designed to process complex relationships in financial data. It uses tensor operations to model interactions between input features.

**Key Components:**

- **Custom Layer (TNet):**
  - Defines a tensor layer (`tlayer`) that performs complex transformations on the input data.
  - Incorporates batch normalization for better training stability.

- **Regularization:**
  - Uses L2 regularization to prevent overfitting.

- **Model Architecture:**
  - The model consists of multiple input layers for different parts of the data.
  - Outer and inner branches are calculated using tensor layers to capture high-level interactions.
  - Combines the outputs from these branches to form the final prediction.

- **Loss Function:**
  - Custom `margin_loss` function designed to handle the specific needs of the TNet architecture.
  - Ensures the model learns to distinguish between correct and corrupted examples effectively.

**Training Process:**

- Utilizes Adam optimizer for training.
- Incorporates a custom data generator (`Generator` class) to yield batches of data.
- Uses ModelCheckpoint to save model weights during training.

## Usage

### CNN

```bash
python cnn.py train <ticker>
```

```bash
python cnn.py predict <ticker> <weight>
```

```bash
python tnet.py train
```

```bash
python tnet.py predict <weight>
```

## Contact

Created by Thomas F McGeehan V

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
