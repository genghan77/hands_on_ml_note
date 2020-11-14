# Decision Trees

Deicison trees don't require feature slcaing or centering at all.

**Algorithm**

* Training and visualizing
    ```python
    three_clf.fit(X, y)
    export_graphviz(
        tree_clf,
        out_file = image_path("tree.dot"),
        feature_names = iris.feature_names[2:],
        class_names = iris.target_names.
        rounded = True,
        filled = True
    )
    ```
    ```bash
    # converting .dot file to png file
    $ dot -Tpng tree.dot -o tree.png
    ```

* Gini impurity or entropy

  * Gini impurity (default)
    $$
    G_i = 1 - \sum{p_{i,k}^2}
    $$
    $p_{i, k}$ is the ratio of class k instances among the training instances in the ith node
    
  * Entropy

    ```python
    criterion = "entropy"
    ```
    $$
    H_i = - \sum{p_{i,k} \times log_2{p_{i,k}}}
    $$
  Gini is slightly faster to compute, so it is a dafuault. Gini tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.

* Note
  
  * sklearn uses CART algo, which produces on binary trees. Other algos like ID3 can produce trees with nodes that have more than two children.

* Estimating probability 
  
  Tree does so by first traverses to find the leaf node for this instance, and then returns the ratio of training instance of class k in this node. 

* CART is a greedy algorithm. It often produces a reasonaly good solution, but not guaranteed to be the optimal. 

* Computational complexity
  
  * Inference takes O(log(m)). Training takes O(nxmlog(m)).sklearn can speed up training by presorting the data (set presort = True), but this slows down training for larger training sets.

  
* Regularization 

  Decision trees make very few assumptions about the training data. It is often called a nonparametric model, becuase the nubmer of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a parametric model has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting. 

  * Ristriction

    Increasing min_* or reducing max_* will regularize the model

    ```python
    max_depth = 3 # default is None, which means unlimited
    min_samples_split = 3 # minimum number of samples a node must have before it can split
    min_samples_leaf = 3 # minimal number of samples a leaf node must have
    min_weight_fraction_leaf = 0.3 # similar to min_samples_leaf but expressed in fraction
    max_leaf_nodes = 12 # maximum number of leaf nodes
    max_features = 2 # maximum number of features that are evluated for splitting at each node
    ```
  * Pruning
    
    A node whose children are all leaf nodes is considered unnecessary if the purity improviement it provdes is not statistically significant. For example, Chi-test are used to estimate the probability that the imrpvoement is purely the result of chance. And if the p-value is bigger than a given threshold, typically 5%, then the node is considered unnecessary. 

* Regression
    
  The predicted value is always the average target value of the instances in that region. For CART algorithm, regression works almost the same way as classification, except that instead of trying to split the training set in a way that minimizes impurity, it now tries to split the training set in a way that minimizes the MSE.


**Instability**

* It prefers orthogonal decision boundaries, which makes them senstivie to training set rotation. One way to limit this problem is to use PCA, which often results in a better orientation of the training data. 

* Generally, trees are very sensitive to small variations in training data. And the implmenetation in sklearn is stochastic, it may lead to very different models even on the same training data, unless random_state hyper parameter is set. Random forest can limit this instability by averaging  predictions over many trees.
