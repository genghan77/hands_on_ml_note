# Training models

In ML, vectors are often represented as *column vectors*, which are 2D arrays with a single column. If $\theta$ and $X$ are column vectors, then the prediction is $\hat{y} = \theta^TX$

**Normal equation**

To find the value of $\theta$ that minimizes the cost function, there is a *closed-form solution*, which is defined as below:
$$
\hat{\theta} = (X^TX)^{-1}X^Ty
$$

```python
theta_best = np.linalg.inv(X_b.T.dot(X+b)).dot(X_b.T).dot(y)

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new] # add x0=1 to each instance
y_predict = X_new_b.dot(theta_best)



#Below use sklearn linear regression
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_

# LinearRegression class in sklearn is based on the scipy.linalg.lstsq() which you can call directly as below:
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
# The above function computes theta_hat = X^{+} where X^{+} is the pseudoinverse of X. You can use np.linalg.pinv() to compute the pseudoinverse directly

np.linalg.pinv(X_b).dot(y)
```

* Note

  pseudoinverse is computed using SVD. And it sets zero to all values smaller than a tinay threshold value and then replaces all the non-zero values with their inverse, and then finally transposes the resulting matrix. It is more efficient than computing the Normal Equation, and it is always defined even if the matrix X^{T}X is not invertible. 

  The computational complexity of inverting a matrix is typically about $O(n^{2.4})$ to $O(n^3)$. And the SVD appraoch using sklearn is usually $O(n^2)$. Both normal euqation and the sVD get very slow when the number of features grow large (100,000). On positive side, both are linear with regards to the nubmer of instances in training set, so they handle large set efficiently, provided they can fit in memory. 


**Gradient descent**

* Note

  The MSE cost function for a Linear REgression model happens to be *convex funciton*, which means there are no local minima, just one global minimum. It is also a continuous function with a slpe that never changes abruptly (Lipschitz continuous). 
 
  When using gradient descent, you should esure that all features have a similar scale, or else it would take much longer to converge.

* Batch gradient descent

  It uses the whole batch of training data at every step. As a result, it is terribly slow on very large training sets. However, it scales well with the number of features, which means that training a linear regression model when there are hundreds of thousands of features is much faster using gradient descent than using normal equation or SVD decomposition. 

  ```python
  eta = 0.1
  n_iter = 1000
  m = 100

  theta = np.random.randn(2,1) # random initialization

  for itertaion in range(n_iter):
    gradient = 2/m*X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta * gradient
  ```

  How to set the numer of iterations? A simple solution is to set a very large number of iterations but to interrupt the algo when the gradient vector becomes tiny - when its norm becomes smaller than a tiny number $\epsilon$. 
  
  It can take $O(1/\epsilon)$ iterations to reach the optimum within a range of $\epsilon$ depending on the shape of the cost function. If you divide the tolerance by 1- to have a more precise solution, then the algo may have to run about 10 times longer. 

* Stochastic gradient descent

  * Compared to batch gradient descent, it makes it possible to handle large dataset, since only one instance needed.

  * And also it is much faster. On the other hand, due to its stochastic nature, it is much less regular than Batch gradient descent. 

  * It has a better chance of finding the global minimum than batch gradeint descnet, since it can actualy help the algo jump out of the local mimima. 

  * TO overcome the issue of never settle at th minimum, one solution is to gradually reduce the leraning rate. This process is akin to *simulated annealing*, and the function that determines the learning rate at each iteration is called the *learning schedule*.

  * When using sgd, the training instances must be IID, to ensure that the paramter get pulled towards the global optimium, on average. A simple way to ensure this is to shuffle the instaces during training. 
  
  ```python
  sgd_reg = SGDRegressor(max_iter=1000, 
    tol=1e-3, penalty=None, eta0=0.1)
  ```

* Mini=batch gradient descent

  Inbetween batch gradescent and sgd. 

* Comparison

  ![image info](../pictures/comparison_lr.png)

**Polynomial regression**

Use sklearn's PolynomialFeatures class to trainsform the trainingdata. 

```python
poly_features = PolynomialFeatures(degree=2,      include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg.fit(X_poly, y)
```

PolynomialFeatures(degree=d) would transforms an array containing n features into an array containing $\frac{(n+d)!}{n!d!}$.

```python
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
```

**Learning curve**

There are two ways to tell whether the model is overfitting or underfitting the data. 

* Use cross-validation. If the model performs well on training set but generalizes poorly according to the cross-validation metrics, then the model is overfitting. If it performs pooly on both, then it is underfitting.

* Look at the *learning curves*. Plot the model's performance on the training set and the validation set as a function of the training set size (or the training iteration), producing something as below:

![image info](../pictures/learning_curve.png)

**Regularized linear models**

It is quite common for the cost function used during training to different from the performance measure used for testing. Apart from regularization, another reason why they might be different is that 

* Ridge regression 

  $$
  J(\theta) = MSE(\theta)+\alpha \frac{1}{2}\sum^n_{i=1}\theta^2_i
  $$
  Note that the bias term $\theta_0$ is not regularized. It is important to sclae the data before performing Ridge Regression, as it is senstivie to the scale of the input features. This is true of most regularized models. 

  The closed-form solution for Ridge Regression is as below:

  $$
  \hat{\theta} = (X^TX+\alpha A)^{-1}X^Ty
  $$

  ```python
  ridge_reg = Ridge(alpha=1, solver="cholesky")

  sgd_reg = SGDRegressor(penalty="l2")
  ```

* Lasso regression

  $$
  J(\theta) = MSE(\theta)+\alpha \sum_{i=1}^n \vert \theta_i \vert
  $$

  Note that lasso tends to completely eliminate the weights of the least important features. 

  ```python
  lasso_reg = Lasso(alpha=0.1)

  sgd_reg = SGDRegressor(penalty="l1")
  ```

* Elastic Net
 
  $$
  J(\theta) = MSE(\theta)+r\alpha \sum_{i=1}^n \vert \theta_i \vert + \frac{1-r}{2}\alpha \frac{1}{2}\sum^n_{i=1}\theta^2_i
  $$

  ```python
  elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5
  )
  ```

* When to use which

  Generally you should avoid plain linear regression. Ridge is a good default. But if you suspect that only a few features are actually useful, you should prefer Lasso or Elastic Net. In general, Elastic Net is preferred over Lasso since Lasso may behave erratically whent he number of eatures is greater than the number of training isntances or when several features are strongly correlated. 

* Early stopping


  ```python
  poly_scaler = Pipeline([
      ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
      ("std_scaler", StandardScaler())
  ])
  X_train_poly_scaled = poly_scale.fit_transform(X_train)
  X_val_poly_scaled = poly_scaler.transform(X_val)

  sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.00005)

  min_val_error = float("inf")
  best_epoch = None
  best_model = None
  for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < min_val_error:
      min_val_error = val_error
      best_epoch = epoch
      best_model = clone(sgd_reg)

  # Note that with warm_start = True, when the fit() method is called, it just continues training where it left off instead of restarting from scratch. 
  ```
  
**Logistic regression**
  
* Estimating probabilities 

  $$
  \hat{p} = h_{\theta}(x)=\sigma(x^T\theta) 
  $$

  The logistic - noted $\sigma()$ - is a sigmoid function that output a number between 0 and 1. 

  Logistic function as below:

  $$
  \sigma(t) = \frac{1}{1+exp(-t)}
  $$
  The score t is often called the logic. The name comes from the fact that the logit function, defined as $logit(p)= log(p/(1-p))$, is the inverse of the logistic function. ANd t is also called the log-odds, which it is the log of the ratio between the estimated probability for the positive clas and the estimated pobability for the negative class.

* Training and cost function 

  $$
  J(\theta) = -\frac{1}{m}\sum^m_{i=1}[y^ilog(\hat{p}^i) + (1-y^i)log(1-\hat{p}^i)]
  $$

  There is no known closed-form equation to compute the value of $\theta$. But the cost function is convex, so Gradient descent is guaranteed to find the global minimum. 

* Decision boundaries

  sklearn actually adds an $l_2$ penalty by default in logistic regression models. The hyperparameter controlling the regularization strength of a sklearn LogisticRegression model is not alpha (as in other linear models), but its inverse: C. The higher the value of C, the less the model is regularized. 

**Softmax regression**

* Cost function

  cross entropy 

  $$
  J(\Theta) = - \frac{1}{m}\sum_{i=1}^m \sum_{k=1}^K y^{i}_k log(\hat{p}^i_k)
  $$

  * Note
    
    If you assumption about the problem of interest is perfect, the cross entropy will just be equal tot he entropy of the problem itself, i.e. the intrinsic unpredictability. But if your assumption is wrong, cross entropy will be greater by an amount called *Kulback-Leibler divergence*. The cross entropy between two probaiblity distribution $p$ and $q$ is defined as $H(p, q) = - \sum_x p(x) log q(x)$, at least when the distributions are discrete. 

  ```python
  softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
  # sklearn's logistic regresion use one vs all by default when you train it on more than two classes. but you can set multi_class hyperparameter to "multinomial" to switch it to softmax regression. You must also specify a solfer that supports softmax regression. It aslso applies $l_2$ regrularization by default, which you can control using C. 
  ```
