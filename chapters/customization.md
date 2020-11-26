# Custom models and training with TensorFlow

**Using tensorflow like numpy**

* Tensors and operations

  ```python
  t = tf.constant([[1., 2., 3.],[4., 5., 6.]])  
  t.shape
  t.dtype

  t[:, 1:]
  t[..., 1, tf.newaxis]
  
  tf.square(t)
  t @ tf.transpose(t) # @ equals to tf.matmul()
  ```

* Tensors and NumPy

  ```python
  a = np.array([2., 4., 5.])
  t = tf.constant(a) # convert numpy to tf tensor
  a = t.numpy() or # np.array(t)

  tf.square(t)
  np.square(a)

  # Notice that NumPy uses 64-bit by default, while tensorflow uses 32-bit. Because 32-bit precision is generally more than enough for nn, plus it runs faster and use less RAM. So when you create a tensor from NumPy array, make sure to set dtype = tf.float32
  ```
* Type conversions

  Type conversion can significantly hurt performance, and they can easily go unnoticed when they are done automatically. You can use tf.cast() to do the conversion if needed. 

  ```python
  t2 = tf.constant(40, dtype=ft.float64)
  tf.constant(2.0) + tf.cast(t2, tf.float32s)
  ```
* Variables 

  ```python
  v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

  v.assign(2*v)
  v[0, 1].assign(42)
  v[:,2].assign([0, 1])
  v.scatter_nd_update(indices=[[0,0],[1,2]], updates=[100., 200.])
  ```
* Other data structures

  * tf.SparseTensor -> contains operation for sparse tensors
  * tf.TensorArray -> they have a fixed size by default, but can optionally be made dynamic. All tensors they contain must have the sam eshape and data type. 
  * tf.RaggedTensor -> static lists of lists of tensors and every tensor has the same shape and data type. 
  * tf.string -> represent byte string, not unicode string. You can represent unicode string using tensors of type tf.int32. 
  * tf.sets
  * tf.queue -> FIFOQueue, PriorityQueue, RandomShuffleQueue, and PaddingFIFOQueue

**Customizing models and training algos**

* Custom loss function

  Take huber loss function for example

  ```python
  # fixed threshold 
  def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

  model.compile(loss=huber_fn, optimizer="nadam")
  model.fit(X_train, y_train, ...)

  # loading model with custom components
  model = keras.models.load_model("my_model_with_a_cusom_loss.h5", custom_objects={"huber_fn": huber_fn}) # Saving a model containing a custom loss function is fine, as keras just save the name of the function. But when you load it, you need to provcide a dictionary that maps the function name to the actualy function.
  ```

  ```python
  # custom threshold
  def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
      error = y_true - y_pred
      is_small_error = tf.abs(error) < threshold
      squared_loss = tf.square(error) / 2
      linear_loss = threshold*tf.abs(error) - threshold**2/2
      return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

  model.compile(loss=create_huber(2.0), optimizer="nadam") # When you save the model, the threshold will not be saved.
  model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5", custom_objects={"huber_fn": create_huber(2.0)}) # Note that the name to use is "huber_fn", which is the anme of the function we gave Keras, not the name of the function that created it. 
  ```

  ```python
  # Custom threshold 

  class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
      self.threshold = threshold
      super().__init__(**kwargs)
    def call(self, y_true, y_pred):
      error = y_true - y_pred
      is_small_error = tf.abs(error) < threshold
      squared_loss = tf.square(error) / 2
      linear_loss = threshold*tf.abs(error) - threshold**2/2
      return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
      base_config = super().get_config()
      return {**base_config, "threshold": self.threshold} # this get_config function make it possible to load the model back without specifying the original threshold
  model.compile(loss=HuberLoss(2.0), optimizer="nadam")
  model = keras.models.load_model("my_model_with_a_custom_loss_class.h5", custom_objects={"HuberLoss": HuberLoss})
  ```

* Custom activation functions, initializers, regularizaers, and constraints

```python
def my_softplus(z): # equivalent to keras.activations.softplus or tf.nn.softplus
  return tf.math.log(tf.exp(z)+1.0)

def my_glorot_initializer(shape, dtype=tf.float32): # equivalent to keras.initializers.glorot_normal
  stddev = tf.sqrt(2/(shape[0]+shape[1]))
  return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights): # equivalent to keras.regularizers.l1(0.01)
  return tf.reduce_sum(tf.abs(0.01*weights))

def my_positive_weights(weights): # equivlanet to tf.nn.relu(wegiths) or keras.constraints.nonneg()
  return tf.where(weights<0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
```

```python
# if the function has some hyperparameters that need to be saved along with the model, then need to subclass the class like for the custom loss above. 
class MyL1Regularizer(keras.regularizers.Regularizer): 
  def __init__(self, factor):
    self.factor = factor def __call__(self, weights):
  return tf.reduce_sum(tf.abs(self.factor * weights)) 
  def get_config(self):
    return {"factor": self.factor}
```
  
You must implement call() for losses, layers, activation functions and models, or __call__() for regularizers, initializers and constraints. 

* Custom metrics

  ```python
  model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
  # For each batch during traning, Keras wil compute this metric and keep track of its mean since the beginning of the epoch. But sometimes this is not what you want. 
  ```

  ```python
  # in the case of binrary classification where you want Precision to be the metric, simply averaging batch results are wrong. You need to keep track of the metric and update from batch to batch. This is called streaming / stateful metric. 
  precision = keras.metrics.Precision()
  p.result() # get the current value of the metric
  p.variables # tracking the number of true and false positives
  p.reset_states() # both variables get reset to 0.0


  # If you need to create one
  class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
      super().__init__(**kwargs)
      self.threshol = threshold
      self.huber_fun = create_huber(threshold)
      self.total = self.add_weight("total", initializer="zeros") # use add_weight to create the varaibles needed to keep traick of metric's state over multiple batches
      self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
      metric = self.huber_fun(y_true, y_pred)
      self.total.assign_add(tf.reduce_sum(metric))
      self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
      return self.total / self.count

    def get_config(self):
      base_config = super().get_config()
      return {**base_config, "threshold": self.threshold}
  ```

* Custom layers

  ```python
  exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x)
  )
  ```

  TO build a custom stateful layer, you need to create a subclass of the keras.layers.Layer class. 

  ```python
  class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
      super().__init__(**kwargs)
      self.units = units
      self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
      self.kernel = self.add_weight("kernel", shape=[batch_input_shape[-1], self.units], initializer="glorot_normal")
      self.bias = self.add_weight("biase", shape=[self.units], initializer="zeros")
      super().build(batch_input_shape)

    def call(self, X):
      return self.activation(X @ self.kernel + self.bias)
    def compute_output_shape(self, batch_input_shape):
      return tf.TensorShape(batch_input_shape.as_list()[:-1], [self.units])

    def get_config(self):
      base_config = super().get_config():
      return {**base_config, "units": self.units,
        "activation": keras.activations.serialize(self.activation)}
  ```

  TO create layer with multiple input and output

  ```python
  class MyMultiLayer(keras.layers.Layer):
    def call(self, X):
      x1, x2 = X # the argument must be a tuple containing all the inputs
      ...
    def compute_output_shape(self, batch_input_shape):
      b1, b2 = batch_input_shape 
      return [b1, b1, b1 ] # output must return the list of batch output shapes one per output
  ```

  To create laeyrs with different behavior during training and testing 

  ```python
  class MyGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
      super().__init__(**kwargs)
      self.stddev = stddev
    def call(self, X, training=None):
      if training:
        noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
        X = X + noise

      return X
  ```

* Custom models 

  ```python
  class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
      super().__init__(**kwargs)
      self.hidden = [keras.layers.Dense(n_neurons, activation="elu", kernel_initializer="he_normal") for _ in range(n_layers)]
    
    def call(self, inputs):
      Z = inputs
      for layer in self.hidden:
        Z = layer(Z)
      return Z + inputs

    class ResidualRegressor(keras.models.Model):
      def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

      def call(self, inputs):
        Z = self.hidden(inputs)
        for _ in range(1+3): 
          Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    # if you wnat to be able to save the model using save() and load it using keras.models.load_model(), you must implement get_config(). Alternatively, you can just save and load the weights using save_weights() and load_weights().
  ```

* Losses and metrics based on model internals 

  The custom losses and metrics we defined earlier were all based on the labels and the predictions. However, sometimes you might want to define loss based on other parts of your model. This myabe useful for regularization purposes, or to minotr some internal aspects of the model. 

  * To define a custom loss based on model internals, just comput it based on the part and pass th eresult to add_loss() method.

  ```python
  class ReconstructionRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
      super().__init__(**kwargs)
      self.hidden = [keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal") for _ in range(5)]
      self.out = keras.layers.Dense(output_dim)
    
    def build(self, batch_input_shape):
      n_inputs = batch_input_shape[-1]
      self.reconstruct = keras.layers.Dense(n_inputs)
      super().build(batch_input_shape)

    def call(self, inputs):
      Z = inputs
      for layer in self.hidden:
        Z = layer(Z)
      reconstrucstion = self.reconstruct(Z)
      recon_loss = tf.reduce_mean(tf.square(reconstrucstion - inputs))
      self.add_loss(0.05* recon_loss)
      return self.out(Z)
  ```


  * To define metric based on model internals, you can compute it on the model part and in call() pass the result to add_metric()

* Computing gradients using autodiff

  ```python
  def f(w1, w2):
    return 3*w1**2 + 2*w1*w2

  w1, w2 = tf.Variable(5.0), tf.Variable(3.0)
  with tf.GradientTape() as tape:
    z = f(w1, w2)
  # gradients context would automatically record every operation that involves a variable and we askthis type to conpute the gradeint 
  gradients = tape.gradient(z, [w1, w2]) # only put strict minimum inside the tf.GradientType() block to save memory. Alternatively, you can pause recording by creating a with type.stop_recording() blaock insidethe tf.Gradeint.Type()

  # This type is automatically erased immediately after you call its gradient() method, so you will get an exception if you try to call gradient() twice. If you need to call gradient() more than once, you must make the taype persistent and delete it when you are done with it to free resources

  with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
  
  dz_dw1 = tape.gradient(z, w1)
  dz_dw2 = tape.gradient(z, w2)

  # By defalt, the taype will only track operation involving variables, if you try to compute the gradient of a non-variable, the result would be none. Hoever, you can force the tape to watch any tensors you like, so that you can compute gradients with regards to these tensors as if they were variables. 
  c1, c2 = tf.constant(5.) , tf.constant(3.)
  with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)

    z = f(c1, c2)

  gradients = tape.gradients(z, [c1, c2])

  # The below would calculate hessians
  with tf.GradientType(persistent=True) as hessian_tape:
    with tf.GradientTpe() as jacobian_tape:
      z = f(w1, w2)
    jacobians = jacobian_tape.gradient(z, [w1, w2])
  hessian = [hessian_tape.gradient(jacobian, [w1, w2]) for jacobian in jacobians]

  del hessian_tape

  # If you want to stop the gradient from backpropagating thru the network, use tf.stop_gradient()

  def f(w1, w2):
    return 3*w1**2+tf.stop_gradient(2*w1*w2)

  with tf.GradientType() as tape:
    z = f(w1, w2)

  gradients = tape.gradient(z, [w1, w2])
  ```

  There are sometimes cases where the autodiff leads to some numerically difficulties. You can tell TF to use stable function when computing the gradients by dicorating it with @tf.custom_gradient, and making it return both its nromal output and the function that computes the derivatives. 

  ```python
  @tf.custom_gradient
  def my_better_softplus(z):
    exp = tf.exp(z)
    def my_softplus_gradient(grad):
      return grad/ (1+1/exp)
    return tf.math.log(exp+1), my_softplus_gradients
  ```