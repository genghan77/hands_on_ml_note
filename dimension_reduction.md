# Dimensionality reduction

**Main approaches for dimensionality reduction**

* Projection
  
  In most real-world problems, training instances are not spread out uniformaly across all dimensions. Many features are almost constant, while others are highly correlated. As a result, all training instances lie within (close to) a much lower-dimensional subspace. So projections works.

* Manifold learning (swiss roll example)

  A d-dimensional manifold is a part of an n-dimensional space that locally resembles a d-dimensional hyperplane (where d<n)

If dimensionality of the trainnign set is reduced before training, it will usually speed up training, but it many not lead to a better or simpler solution. 


**Algorithms**

* PCA

  * Select the axis that preserves the maximum amount of variance, as it will most likely lose less information. And it also minimizes the mean squared distance between the original dataset and its projection. The direction of the principle components is not stable, but the generated PC would still lie on the same axes.
  
  * From scratch
  ```python
  # SVD can be used to obtain the principle components.
  # PCA assumes the dataset is centered around the origin. Sklearn implementation take care of the centering. If PCA is implemented from scratch, then need to center the data first.
  X_centered = X-X.mean(axis = 0)
  U, s, Vt = np.linalg.svd(X_centered)
  c1 = Vt.T[:, 0]
  c2 = Vt.T[:, 1]


  # projecting the data down to the d dimensions
  W2 = Vt.T[:, :2]
  X2D = X_centered.dot(W2)
  ```

  * Sklearn
    
    In sklearn implementation, principle components can be accessed using components_. So the first pc is equal to pca.components_.T[:, 0]
    ```python
    pca.explained_variance_ratio_
    ```

  * Right number of dimensions

    Usually preferable to choose the number of dimensions that add up to a sufficiently large portion of the variance (say 95%), unless it is for DataViz (usually 2, 3 dimensions)

    ```python
    # finding d manually
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum>=0.95) +1

    # automatic
    pca = PCA(n_components=0.95)
    ```

    Elbow method can also be used by plotting cumulative explained variance against dimensions.

  * Reverse transform
    ```python
    # it won't be fully recovered, but fairly close
    X_recovered = pca.inverse_transform(X_reduced)
    ```

  * Randomized PCA

    ```python
    # sklearn use Randomized PCA to quickly find approximation of the first d pc. Dramatically faster. 
    svd_solver = "randomized" 
    
    svd_solver = "full" # use full SVD
    svd_solver = "auto" # default, sklearn will use randomized if m and n is greater than 500, or d is less than 80% of m or n
    ```

  * Incremental PCA
    
    Useful for large training set, also to apply PCA online. 

    ```python
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X_train)
    ```
* Kernel PCA
  
  It is good at preserving clusters of instances after projects, or sometimes even unrolling datasets that lie close to a twisted manifold.

  * Selecting kernal and hyperparameters


    ```python

    # Used as dimensionality method and use cv to choose
    clf = Pipeline([
        ("kpca", KernelPCA(n_componenets = 2)),
        ("log_reg", LogisticRegression())
    ])
    param_grid = [{
        "kpca_gamma": np.linspace(0.03, 0.05,10),
        "kpca_kernel": ["rbf", "sigmoid"]
    }]
    grid_serach = GridSearchCV(clf, param_grid, cv = 3)
    grid_search.best_params_


    # Use reconstruction error to choose

    rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", 
                        gamma=0.001, fit_inverse_transform = True) # By default fit_inverse_transform is False, and kernelPCA has no inverse_transform() method. It only gets created when fit_inverse_transform is set to True

    X_reduced = rbf_pca.fit_transform(X)
    x_preimage = rbf_pca.inverse_transform(X_reduced)
    mean_squared_error(X, X_preimage)
    ```

* LLE (Locally linear embedding)

  It works by first measureing how each instance linearly relates to its closest neibhbors, and then looking for low-dimensional representation of the dataset where these local relationships are best preserved. Works well at unrolling twisted manifolds, especially when there is not much noise. It scale poorly to large datasets.

  ```python
  lle = LocallyLinearEmbedding(n_components = 2, n_neighbors = 10)
  ```

* Other techniques

  * Multidimensional scaling (MDS). Trying to preserve distances bewteen instances.

  * Isomap. Creates a graph by connecting each instance to its nearest neighbors, then reduces dimensionality while trying to preserve the geodesic distances. 

  * tSNE. Keep similar instances close and dissimlar instaces apart. Often use for DataViz

  * LDA. Keep classes as far apart as possible. So is a good technique to be used before running another classificaiton algo, say SVM. 


