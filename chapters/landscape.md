# Landscape

**Types of machine learning problems**

* Note

    In ML, an attribute is a data type (e.g., "Mileage")m while a feature has several meanings depending on the context, but generally means an attrivute plus its value (e.g., "Mileage = 15,000")

* Supervised / unsupervised 
    * Supervised

    Some of the most important supervised learning algo:

    * k-nearest neighbors
    * Linear regression
    * Logistic regression
    * SVM
    * Decision trees and random forests
    * Neural networks

    * Unsupervised learning
    
    * Clustering
        
        * k-means
        * DBSCAN
        * HCA (Hierarchial cluster analysis)
    * Anomaly detection and novelty detection
        * one-class svm
        * isolation forest

    * Visualization and dimensionality reduction
        * PCA
        * kernel pca
        * LLE (locally-linear embedding)
        * tSNE
    * Association rule learning
        
        * Apriori
        * Eclat

    * Semisupervised learning

    * Recinforcement learning

* Batch and online learning

  Whether ml system can learn incrementally from a stream of incoming data

  * Batch learning 

    In batch learning, the system is incapable of learning incrementally, it must be trained using all the available data. 

    Cons:

    * Training the full set of data can take many hours, so would typically train a new system only every 24 hours or even weekly. If your system needs to adapt to rapidly changing data (to predict stock prices), then you need a more reactive solution. 

    * Training on full set requres a lot computing resources. If you have a lot data and automate the system to train from scratch everyday, it would end up costing a lot of money. And if the amoung of data is huge, it may even be impossible to use a batch learning algo. 

    * If your system needs to be able to learn autonomously and has limited resources, then carrying around large amount of data and taking up a lot resources to train for hours everyday is a showstopper. 

  * Online learning
  
    Online learning is great for systems that receive data as a continous flow and need to adapt to change rapidly or autonomously. It is also a good option if you have limited computing resources. Online learning algos can also be sued to train systems on huge datasets that can not fit in one machine's main memory (*out-of-core* learning), in this manner, online learning can be thought as incremental learning becuase out-of-core learning is usually done offiline. 

    * Learning-rate
      
      How fast should online learning systems adapt to changing data. If it is set high, the system would rapidly adapt to new data, but tend to quickly forget the old ones. On the other hand, if set a low value, the system would have more inertia. 

    * What if bad data is fed to the system ? 
      
      * Monitor closely and promptly switch learning off and possily revert to a previously working state if you detect a drop in performance. 

      * Monitor the input data and react to abnormal data e.g. using an anomaly detection algo. 

* Instance-based versus model-based learning
  
  Two main approaches to generalization.

  * Instance-based learning

    The system learns the examples by heart, then generalizes to new cases by comapring them to the learned exmaples or a subset of them, using a similarity measure. 

  * Model-based learning

    Study the data -> select a model -> train it on training data -> apply the model to make predictions on new cases. 

**Challenges**

* Insufficient quantity of training data 

* Nonrepresentative training data 

  If the sample is too small, you will have sampling noise (i.e. non representative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias. 
    
* Poor quality data

  Training data with errors, outliers, and noise (due to poor-quality measurements). 

  * If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually. 

  * If some instances are missing a few features (say 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values, or train one model with the feature and one model withoutit. 

* Irrelavant features

  Feature engineering is needed :
 
  * Feature selection. 
  * Feature extraction: combining existing features to produce a more useful one. 
  * Creating new features by gathering new data.

* Overfitting the training data
* Underfitting the training data
  
**Testing and validating**

* Hyperparameter tuning and model selection

  * Hold out part of the training set to evaluate aseveral candidate models, and select the best one. More specifically, you train multiple models with various hyperparameters on the reduced training set, and select the model that performs best on the validation set. And after this process, you train the best model ont eh full training set, and this gives you the final model. lastly, evaluate this final model on the test set to get an estimate of the generatlization error. 

  * An alternative is to perform repeated cross-validation, using many small validation sets. Each model is evaluated once per validation set, after trained on rest of the data. By averaging out all the evaluatio of a model, we get a much more accurate measure of its performance. The drawback is, the training time is multiplied by the number of validation sets. 

* Data mismatch

  Say you build an app for flower recognition from mobile app, and your training set are the flower images from the web. The most important rule to reember si that the validation set and the test set must be as representative as possible fo the data you expect to use in production, so they should be composed exclusively of representative pictures: shuffle and put half in validation set, and half in test set, and make sure that no duplicates or near-duplicates end up in both sets. 

  After training the model on web pictures, if the performance of the model on the validation set is disappointing, you would not know wehther this is because ur model overfit or due to mismatch. One solution is to use train-dev set, which is taken from trainign set. So train the model on reduced train set, and evaluate on train-dev set, if it performs wellm then the issue is data mismatch, if otherwise, then it should be model overfit to training data. 

  
    
