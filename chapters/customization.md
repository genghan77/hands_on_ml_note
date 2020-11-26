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