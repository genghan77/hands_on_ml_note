# Loading and preprocessing data with TF

**Data API**

The  whole Data API revolves aroundt he concept of a *dataset*. 

* Chaining transformation

  ```python
  X = tf.range(10)
  dataset = tf.data.Dataset.from_tensor_slices(X)
  tf.data.dataset.range(10) # equivalent to the above

  dataset = dataset.repeat(3).batch(7) # you can drop the last batch so that every batch has the same size by setting drop_remainder=True

  # To apply transformation on individual item
  dataset = dataset.map(lambda x: x*2) # you can spawn multiple threads by setting num_parallel_calls arguments

  # TO apply transformation on the total dataset as a whole you can use apply()
  dataset = dataset.apply(tf.data.experimental.unbatch())

  dataset = dataset.filter(lambda x:x<10)
  dataset.take(3)
  ```

* Shuffling the data

  Gradient descent works best when the instances in the training set are independent and identically distributed. One way to ensure this is to shuffle the instances. For this, you can just use the shuffle() method. 

  ```python
  dataset = dataset.shuffle(buffer_size=5, seed=42) # call repeat() on a shuffled dataset will generate a new order at every iteration. This is generally a good idea, but if you prefer to reuse the same order at each iteration, you can set reshuffle_each_iteration=False
  ```
  * Interleaving lines from multiple files

    ```python
    train_filepaths = [, ] # contains a list of file paths
    filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42) # by default, the list_files() function returns a datset that shuffles the file paths. In general, this is a good thing, but you can set shuffle=False if you don't want to. 

    n_readers = 5
    dataset = filepath_dataset.interleave(
        lambda filepath: tf.data.TextLinedatset(filepath).skip(1),
        cycle_length=n_readers
    ) # create a dataset that will pull 5 file paths from filepath_dataset. By default, interleave() does not have paramllelism. you can set num_parallel_calls to the number of threads you want. you can even set tf.data.experimental.AUTOTUNE to make TF choose for the right number of threads dynamically. 
    ```
* Preprocessing the data

  ```python
  X_mean, X_std = [...] # pre-computed, 1d tensors containing 8 floats, one for each feature
  n_inputs = 8

  def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)] # missing value in feature columns are default to 0 and they are floats. Target is floats, and there is no default value, if it encoutners a missing value it would raise an exception. 
    fields = tf.io.decode_csv(line, record_defaults=defs) # line is the line to parse, the second is an array containing the default value for each column in the csv file. This returns a list of sclaer tensors (one per column)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x-X_mean)/X_std, y
  ```
  * Prefetching 

    By calling prefetch(1) at the end, we are creating a dataset that will do its best to always be one batch ahead.  
  
**TFRecord Format**

TFRecord format is TF prefer format for handling large amounts of data. 

**Preprocessing the input feautres**

* You can either encode categorical features using one-hot or embedding. As a rule of thumb, if the number of categories is lower than 10, then one-hot is generally the way to go. If it is greater than 50, then embedding are preferable. In betwee, you may want to experiment and see which one works best. 
  
* For encoding categoical features using embedding, as a rule of thumb, embeddings typically have 10 to 300 dimensions, depending on the task and the vocab size you have. 

* Keras preprocessing layers

  * One example would be Discretization layer. It is not differentiable, and it should only be used at the start of your model. You should not use an Embedding layer directly in a custom preprocessing layer, if you want it to be trainable. Because the model's preprocessing layers will be forezen during taning. In this case, if you want it to be trainable, you need to add it separately to the model. 

  * If the standard preprocessing layers are insifficient, you have the options to create your own preprocessing layer. Create a subclass of the keras.layers.PreprocessingLayer class with an adapt() method, which should take a data_sample argument and optionally an extra reset_state argument. 

**TF Transform**

