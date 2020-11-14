# Ensemble learning and random forests

**Voting**

Aggregate the predictions of each classifier (diverse) and predict the class that gets the most votes, named hard voting classifier. 

* Note

  Ensemble methods work best when the predictors are independent from one another as possibile. One way to get diverse classifiers is to train them using very different algorithms. 

* Receipe

  ```python
  # To change to soft voting
  # 1. voting = "soft"
  # 2. make sure all classifier has predct_proba()
  # for case of svm, by changing probability = True would make svc use cross-validation to estimate probability, slowing down training, and it will add a predict_proba method.
  voting_clf = VotingClassifier(
      estimators = [('lr', log_clf),('rf', rnd_clf),('svc', svm_clf)],
      voting = "hard" # change to soft will use soft voting
  )
  voting_clf.fit(X_train, y_train)

  for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predct(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
  ```

  Soft voting often achives higher performance than hard voting becasue it gives more weight to highly confident votes. 

**Bagging and pasting**

Besides using very different training algoritms, another approach to get a diverse set of classifiers are to use the same training algorithm, but train them on different random subsets of the training set. When sampling is done with replacement, it is called bagging, otherwise pasting. 

The aggregation function is typically the statistical mode for classification (most frequent prediction, just like a hard voting classifier), or average for regression. Generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained on the original training set. 

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators = 500,
    max_samples = 100, bootstrap = True, n_jobs = -1
)

# If bootstrap = False, it would be pasting
# Bagging classifier automatically performs soft boting if the base classifier can estimate class probabilities.
```

Boostrapping introduces a bit more diversity in the subsets, so bagging ends up with a lightly higher bias than pasting, but this also means that predictors end up being less correlated so the variance is reduced. Overall, bagging usually results in better models, and it is generally preferred. 

**Out of bag evaluation**

By default, a Bagging Classifier samples m training instances with replacement (boostrap = True) where m is the size of the training set. This means that only about 63% of the training instances are sampled on average, and the remaining 37% of the training instances that are not sampled are called out-of-bag instance. Note that they are not the same 37% for all predictors. 

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators = 500,
    bootstrap = True, n_jobs = -1, oob_score = True
)
bag_clf.oob_score_
bag_clf.oob_decision_function_ # oob decision function for each training instance
# by setting oob_score = True, an automatic oob evaluation after training is requested. 
```

**Random patches and random subspace**

Random patches: sampling both training instances and features 

Random Subspaces: sampling just the features and keeping all training instances (boostrap = False, max_samples = 1.0, bootstrap_features = True, max_features = 0.5)

Sampling features results in even more predictor diversity trading a bit more bias for a lower variance. 

**Random forests and Extra-trees**

* Random forests

  Random forest is an ensemble of decision trees. Instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features, and this results in greater tree diversity, which trades a higher bias for a lower variance, generally yielding an overall better model. 

  ```python
  rnd_clf = RandomForestClassifier(n_estimator = 500, max_leaf_nodes = 16, n_jobs = -1)
  # It would be roughly the same as putting deicision tree classifier into bagging classifier and setting splitter = "random"
  ```

* Extra-trees

  In random forest, at each node, only a random subset of the features is considered for splitting, and it is possible to make trees even more randm by also using random thresholds for each feature rather than searching for the best possible thresholds. This is an Extremely Randomized Tree ensemble, or Extra-trees for short. It trades more bias for a lower variance. It makes the ensemble much faster to train. 

* Which one to choose

  It is hard to tell in advance which one would perform better, the only way to know is to try both and com pare them using cross-validation (and tuning hyperparameters using grid search).

**Feature importance**

sklearn measures a feature's importance by looking at how much the tree nodes that use that feature reduce impurity on average (across all trees in the forest). More precisely, it is a weighted average, where each nodes' weight is equal to the number of training samples that are associated with it. 

```python
for name, score in zip(iris["feature_names"], rnd_clf.feature_importantces_):
    print(name, score)
```

Random forests are very handy to get a quick understanding of what features actually matter, in particular if feature selection is needed. 

**Boosting**

The general idea of most boosting methods is to train preditors sequentially, each training to corre t its predecessor. The most popular are Adaboost and Gradient boosting. 

* Adaboost

  Let new predictor to correct its predecessor by paying a bit more attention to the training isntances that the predecessor underfitted. This results in new predictors focusing more and more on the hard cases. 

  * Note

    learning_rate refers to the weights that are boosted at each iteration. So compare learning_rate = 1 to 0.5, the misclassified instance weights are boosted twice as much. 

    One important drawback is that it can not be parallelized, or only partially. As a result, it does not scale as well as bagging or pasting. 

  Once all predictors are trained, the ensemble makes predictions very much like bagging or pasting, except that predictors have different weights depending on their overall accuracy on the weighted training set. 

  * Receipe

    ```python
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassification(max_depth=1), n_estimators = 200,
        algorithm = "SAMME.R",
        learning_rate = 0.5
    )
    # sklearn use SAMME. When there are just two classes, SAMME is equivalent to Adaboost. Moreover, if the predictors can estimate class probabilities, then sklearn can use SAMME.R , which relies on class probabilities rather than predictions and generally performs better. 
    # if AdaBoost is overfitting, we can try reducing the number of estimator or more strongly regularize the base estimator
    ```

  * Algorithm 
  
  Weighted error rate of the $j^{th}$ predictor 
  $$
  r_j = \frac{\sum_{y_j^i \neq y^i}{w^i}}{\sum{w^i}}
  $$
  where $y_j^i$ is the $j^{th}$ predictor's prediction for the $i^{th}$ instance.

  Predictor weight 
  $$
  \alpha_j = \eta log \frac{1-r_j}{r_j}
  $$

  Weight update rule 
  
  for $i = 1, 2, ..., m$
  $$
  w^i = \begin{cases}
      w^i, & \text{if}\ y_j^i = y^i \\
      w^i exp(\alpha_j), & \text{otherwise}
    \end{cases}
  $$
  Then all instances weights are normalized. 

  The algorithm stops when the desired number of predictors is reached, or when a perfect predictor is found. 

  Predictions

  To make predictions, AdaBoost simply computes the predictions of all the predictors and weights them using the predictor weights $\alpha_j$. 

  $$
  y(x) = argmax_k \sum{\alpha_j} 
  $$


* Gradient boosting 

  Instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor. 

  ```python
  gbrt = GradientBoostingRegressor(
      max_depth = 2,
      n_estimators = 3,
      learning_rate = 1.0
  )
  # learning_rate scales the contribution of each tree. If you set it to a low value, then it will need more trees in the ensemlbe to fit the training set, but the predictions will usually generalize better. This is called shrinkage.
  ```

  * Finding the optimal number of trees

    * use staged_predict() method

      ```python
      errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]

      bst_n_estimators = np.argmin(errors)
      gbrt_best = GradientBoostingRegressor(max_depth = 2, 
      n_estimators = bst_n_estimators)
      ```

    * mannual early stopping

      ```python 
      min_val_error = float("inf")
      error_going_up = 0
      for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators 
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
          min_val_error = val_error
          error_going_up = 0
        else:
          error_going_up += 1
          if error_going_up == 5:
            break
      ```
  * Note
    
    GradientBoostingRegressor also supports a subsample hyperparameter, which specifies the fraction of training instances to be used for training each tree. This trades a higher bias for a lower variance. And it also speeds up training considerably. This is called Stochastic Gradient Boosting. 

    It is also possible to use Gradient boosting with other cost functions, and this is controlled by the loss hyperparameter. 

  * xgboost

    ```python 
    xgb_reg.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)], 
        early_stopping_rounds = 2
    )
    ```

**Stacking**

The idea is to train a model to perform the aggregation. Each of the base predictor predicts a value, and then the final predictor, called blender or meta learner, take these predictions as input and makes the final prediction.

It is possible to train several different blenders. 

sklearn does not support stacking directly, but open source implmenetation such as [brew](https://github.com/viisar/brew) is available. 


