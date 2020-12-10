# Deep computer vision using convolutional neural networks

* Tricks
  
  * A common mistake is to use conv kernels that are too large. Instead of using a conv layer with 5 x 5 kernel, stack two layers with 3 x 3 kernel is better. One exception is for the first conv layer, it can typially have a large kernel (e.g. 5 x 5) usually with a stride of 2 or more. 

  * The number of filters grous as we climb up the cnn toward the output layer. It is a common practice to double the number of filters after each pooling layer. 
