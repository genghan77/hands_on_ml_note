# Processing sequences using RNN and CNN

* Training for time series

    When dealing with time series, the input features are generally represented as 3D arrays of shape [batch size, time steps, dimensionality], where dimensionality is 1 for univariate time series and more for multi-variate time series. 

    ```python
    # simple RNN
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(1, input_shape=[None, 1])
    ]) 
    # input_shape=[None, 1] the First value is None because RNN can process any number of time steps. 

    # By default, RNN layers in keras only return the final output. To make them return one output per time step, you must set return_sequences=True.

    # predict 10 values at the last time step
    # sequence-to-vector
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.simpleRNN(20),
        keras.layers.Dense(10)
    ])
    # The last layer change to dense so that we can use any activation function as appropriate.

    # predict 10 values at each and every time step. 
    # sequence-to-sequence
    model = keras.models.Sequential([
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
    # TimeDistributed layer make the inner layer apply independently to the input sequence at every time step.

    # When forecasting time series, it is often useful to have some error bars along with predictions. For this, an efficient technique is MC dropout.
    ```

    When using other models to forecast time series, such as *weighted moving average* or *ARMIA (autoregressive integrated moving average)*, one need to remove the seasonality to train and model, and adding the seasonality back when making predictions. 

* Handling long sequences

  * Unstable gradient

    many of the tricks we used in deep nets to alleviate the unstable gradient can help, such as good parameter initialization, faster optimizers, dropout, and so on. But nonsaturating activation function may not help as much, in fact, they may actually lead the RNN to be even more unstable during training. You can reduce the risk by using a smaller learning rate, but you can also simply use a saturating activation function like the hyperbolic tangent (which is the default). If you notice that training is unstable, you may want to monitor the isze of the gradeints using TensorBoard and perhaps use Gradient Clipping. 

    Batch Normalization can not be used as efficiently with RNN as with deep nets. You can't use it between time steps, only between layers. A paper suggest that BN was slightly benefitial when it was applied to the inputs, not to the hidden states. In keras, you can do it by adding a BN layer before each rnn layer, but don't expect too much. 

    Anohter form of normalization often works better with RNN is Layer Normalization. 

    ```python
    class LNSimpleRNNCell(keras.layers.Layer):
        def __init__(self, units, activation="tanh", **kwargs):
            super().__init__(**kwargs)
            self.state_size = units
            self.output_size = units
            self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,activation=None)
                        self.layer_norm = keras.layers.LayerNormalization()
            self.activation = keras.activations.get(activation) 
        
        def call(self, inputs, states):
            outputs, new_states = self.simple_rnn_cell(inputs, states) norm_outputs = self.activation(self.layer_norm(outputs)) return norm_outputs, [norm_outputs]
    ```
  * Short-term memory

    * LSTM

    ```python
      # Both are fine, but lstm layer are optimized. RNN is most useful when you define custom cells.
        model = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))        
        ])

        model = keras.models.Sequential([
        keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True,
                         input_shape=[None, 1]),
        keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
    ```
  * GRU

  * Use 1d cnn to reduce the sequence length

  ```python
  
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    history = model.fit(X_train, Y_train[:, 3::2], epochs=20,
                        validation_data=(X_valid, Y_valid[:, 3::2]))

  # We must crop off the first tree time steps in the targets, since the kernel's size is 4, the first output of the convolutional layer will be based on the input time steps 0 to 3.
  ```

  * WaveNet

  ```python

    model = keras.models.Sequential() model.add(keras.layers.InputLayer(input_shape=[None, 1])) forratein(1,2,4,8)*2:
         model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal", # causal padding ensures the conv layer does not peek into the future when making predictions. It is equivalent to padding the inputs with the right amount of zeros on the left and using "valid" padding
        activation="relu", dilation_rate=rate))
     model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
     model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
     history = model.fit(X_train, Y_train, epochs=20,
                         validation_data=(X_valid, Y_valid))
  ```
    