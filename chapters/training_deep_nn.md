# Tranining deep neural networks

**Vanishing/Exploding gradients**

With sigmoid activation function and random initialization using a normal distribution with a mean of 0 and a standard deviation of 1, the variance of the outputs of each layers is much greater than the variance of its inputs. Going forward in the network, the variance keeps increasing after each layer until the activation function saturates at the top layers. This is made worse by the fact that the logistic function has a mean of 0.5. 

* Glorot and He initialization

  Normal distribution with mean 0 and variance

  $$
  \sigma^2 = \frac{1}{fan_{avg}}
  $$

  or a uniform distribution between $-r$ and $+r$ where

  $$
  r = \sqrt{\frac{3}{fan_{avg}}}
  $$

  where $fan_{avg} = (fan_{in}+fan_{out})/2$ and $fan_{in}$ represents the number of inputs, and $fan_{out}$ represnets the number of neurons.

  If you replace $fan_{avg}$ with $fan_{in}$, you get an initialization strategy that was proposed by Yann LeCun called *LeCun initialization*.


  ![Image info](../pictures/initialization.png)
  These strategies differ only by the scale of the variance and whether they use $fan_{avg}$ or $fan_{in}$. For the uniform distribution, just compute $r=\sqrt{3\sigma^2}$. The initialization strategy for the ReLU and its variance including ELU is someitimes called He initialization. The SELU activation function should be used with LeCun initialization (preferably with a normal distribution).

  ```python
  # By default, keras use Glorot initialization with a uniform distribution, you can change this to he initialization by setting kernel_initializer="he_uniform" or kernel_initializer="he_normal" liike below
  keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

  # If you want He initialization with a uniform distribution, but based on fan_vag rather than fan_in, then you can use the VarianceScaling initializer:
  he_avg_init = keras.initializers.VarianceScaling(scale=2, mode="fan_avg", distribution="uniform")
  keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
  ```

* Nonsaturating activation functions

  * ReLU 
    
    During training, some neurons effectively die. In some cases, you may find that half of your network's neurons are dead, especially if you used a large learning riate. 

  * Leaky ReLU:
    
    $$
    \text{LeakyReLU}_{\alpha}(z) = max(\alpha z, z)
    $$

    this small slope $\alpha$ make sure that they have a chance to eventually wake up. 
    * Papers show that setting $\alpha=0.2$ (huge leak) seemed to result in better performane than $\alpha=0.01$ (small leak). 
    * They also evaluated randomized leaky relu, where $\alpha$ is picked randomly in a given range during training, and it is fixed to an average value during test. It performed fairly well and seemed to act as a regularizer. 
    * Parametric leaky relu, where $\alpha$ is authorized to be learned during training. This report to strongly outperform ReLU on large image datasets, but on smaller datasets it runs the risk of overfitting. 
  
  * ELU
    
    $$
    \text{ELU}_{\alpha}(z) = \{\begin{array}{lr}
        \alpha (exp(z)-1), & \text{if } z<0\\
        z, & \text{if } z \geq 0 
        \end{array}

    $$

    * $z$ is usualy set to 1, but you can tweak it. 
    * It has a nonzero gradient for $z<0$, avoids the dead neurons problem
    * Main drawback is ti is slower to compute, but it is compensated by the faster convergence rate. 

  * SELU
    
    If you build a network composed exclusively of a stack of dense layers, and if all hidden layers sue the SELU activation (scaled version of ELU), the network will self-normalize. 

    * Input features must be standardized.
    * Every hidden layer's weights must be initalized using LeCun normal initialization. In keras, it means setting kernel_initializer="lecun_normal".
    * The network's architecture msut be sequential (no RNN, no skip connection).
    * Self-normalization are only guaranteed when all layers are dense. However, in practice, it seems to work great with CNN.

  * which one to choose?

    SELU > ELU > leaky ReLU (variants) > ReLU > tanh > sigmoid

    * if the networks' architecture prevents it from self-normalizing, then ELu may perform better than SELU. 
    * If you cares about runtime latency, then you may prefer leaky ReLU. 
    * If you don't want to tweak another hyperparemeter, then you may just set deafult $\alpha$ used by keras (0.3). 
    * If you have spare time and computing power, you can use cross-validation to evaluate other activation functions, in particular RReLU if your network is overfitting, or PReLU if you have a huge trainingset. 

    ```python
    # use leaky ReLU
    leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
    layer = keras.layers.Dense(10, activation=leakyReLU, kernel_initializer="he_normal")

    # use PReLU, just replace LeakyReLU with PReLU. 

    # There is currently no official implementation of RReLU. 

    # SELU
    layer = keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")
    ```

* Batch normalization

  $$
  \mu_B=\frac{1}{m_B}\sum^{m_B}_{i=1}x^i 
  $$

    $\mu_B$ is the vector of input means, evaluated over teh whole mini-batch B (contains one mean per input)
    
  $$
  \sigma_B^2 = \frac{1}{m_B} \sum^{m_B}_{i=1}(x^i-\mu_B)^2
  $$
    $\sigma_B$ is the vecotr of input standard deviation evaluated over the whole mini-batch B (contains one std per input).
  $$
  \hat{x^i} = \frac{x^i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
  $$
    $\hat{x^i}$ is the vector of zero-centered and normalized inputs for instance i.

  $$
  z^i = \gamma \otimes \hat{x^i} + \beta
  $$
    $\gamma$ is the output scale parameter vector fot he layer (contains one scale paremeter per input).

    $\otimes$ represents element-wise multiplication (each input is multiplied by its corresponding output scale parameter)
  
    $\beta$ is the output shift parameter vector for the layer (contains one offset parameter per input)

    $\epsilon$ is a tiny number to avoid division by zero. Typically $10^{-5}$, called *smoothing term*

    $z^i$ is the output of the BN operation, it is a rescaled and shfited version of the inputs. 

  Four parameters were learnt during the training process. $\gamma$ (the output scale vector), and $\beta$ (the output offset vector), and $\mu$ and $\sigma$. Note that $\gamma$ and $\beta$ are learnt thru backpropagation, and $\mu$ and $\sigma$ are estimated using an exponential moving average. And $\mu$ and $\sigma$ are estimated during training but used after training. 

  There is a run time penalty for using BN. So if you need predictions to be lightning-fast, you may want to check how well plain ELU + He initialization perform before applying BN.

  ```python
  # Add BN after each hidden layer's activation, and optinally add a BN layer as well as the first layer in your model so that you odn't ahve to standardize the input.

  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Batchnormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Batchnormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Batchnormalization(),
    keras.layers.Dense(10, activation="softmax")
  ])
  
  [(var.name, var.trainable) for var in model.layers[1].variables]
  model.layers[1].updates

  # The BN can also be add before activation. And since BN includes a offset parameter per input, you can remove the bias term from previous layer by setting use_bias=False. There are debates on whether BN should be before or after activation, but it depends on the dataset. 
  
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False)
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False)
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax")
  ])
  ```

  * Hyperparameters for BN

    * momentum
      
      It is used when updating the exponential moving averages. Given a new value $v$, the running average $\hat{v}$ is updated using the following equation:

      $$
      \hat{v} \leftarrow \hat{v} \times \text{ momentum} + v \times (1-\text{momentum})
      $$

      A good momentum value is typically clsoe to 1 --- for exmaple, 0.9, 0.99, 0.999 (you want more 9s for large datasets and smaller mini-batches).

    * axis
      
      It determines which axis should be normalized. It defaults to -1, meaning that by default it will normalize the last axis. 

* Gradient clipping

  Since BN is tricky to use in RNN, gradient clipping is often used. In keras, implementing Gradient Clipping is just a matter of setting the clipvalue or clipnorm argument when creating an optimizer. 

  ```python

  optimizer = keras.optimizers.SGD(clipvalue=1.0)
  model.compile(loss="mse", topimizer=optimizer)
  # The above code snippet clip every component of the gradient vector to a value between -1 to 1. This may change the orientation of the gradeint vector. In practice, this approach works well. Anthoer way is to use clipnorm. This will clip the whole gradient if its $l_2$ norm is greater than the threshold you picked, and reserving tis orientation. 
  ```

  If you observe gradients explode during training (you can track the size of the gradeints using TensorBoard), you may try both clipping by value and norm, with different threshold. 

**Reusing pretrained layers**

