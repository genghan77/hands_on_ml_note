# End to end project

**Look at the big picture**

* Frame the problem

  * What is the business objective ? 

  * What does the current solution looks like ? 

* Select a performance measure

  * Distance measures
    
    The $l_k$ norm of a vector v containing n elements is defined as $\vert \vert v \vert \vert_k = (\vert v_0\vert^k + ... + \vert v_n\vert^k)^{\frac{1}{k}}$. $l_0$ just gives the number of non-zero elements in the vector, and $l_{\infty}$ gives the maximum absolute value in the vector. 

    The higher the norm index, the more it focuses on large values and neglects small ones. That is why the RMSE is more sensitive to outliers than MAE. When outliers are exponentially rare (like in a bell-shape curve), the RMSE performs very well and is generally preferred. 

  * Notes from [Comparison between MAE and RMSE](https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d)
    
    * Similarities

      Both MAE and RMSE express aveage model prediction error in units of the variable of interest. Both range from 0 to $\infty$ and are indifferent to the direction of errors. They are negatively-oriented scores, the lower the better.

    * Differences 
     
      Since errors are squred before they are averaged, the RMSE gives a relatively high weight to large errors. This means the RMSE should be more useful when large errors are particularly undesirable. RMSE does not necessarily increase with the variance of teh errors. RMSE increases with the variance of the frequency distribution of error magnitudes. 

      $$
      [MAE] \leq [RMSE]
      $$

      The RMSE result will always be larger or equal to the MAE. If all the errors have the same magnitude, then $RMSE = MAE$.

      $$
      [RMSE] \leq [MAE * sqrt(n)]
      $$
      where $n$ is the number of test samples. The difference between RMSE and AME is greatest when all of the prediction error comes form a single test sample. 

      RMSE has a tendency to be increasingly larger than AME as the test sample size increases. And this can be problematic when comparing RMSE results calcualted on different sized test samples. 

      RMSE has the benefit of penalizing large errors more. From an interpretation standpoint, MAE is the winner. On the other hand, RMSE is desirable in many mathematical calcualtion. 

* Check the assumptions

  

**Get the data**

* Create the workspace
 
  ```bash
  export ML_PATH = "$HOME/ml"
  mkdir -p $ML_PATH

  python3 -m pip --version
  python3 -m pip install --user -U pip

  python3 -m pip install --user -U virtualenv
  cd $ML_PATH
  virtualenv env

  cd $ML_PATH
  source env/bin/activate # on Linus or MacOSX
  .\env\Scripts\activate # on Windows
  ```
* Download the data

  In typical env, your data would be available in a relational database and spread across multiple tables/documents/files. You would need to familiarize yourself with the data schema. It would be good to write it in a function and setup a scheduled job to do that automatically at regular intervals. 


  ```python
  DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/" 
  HOUSING_PATH = os.path.join("datasets", "housing")
  HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
  def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): 
    if not os.path.isdir(housing_path):
      os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz") 
    urllib.request.urlretrieve(housing_url, tgz_path) housing_tgz = tarfile.open(tgz_path) 
    housing_tgz.extractall(path=housing_path) 
    housing_tgz.close()
  ```

  ```python
  def load_housing_data(housing_path=HOUSING_PATH):      
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)
  ```
* Take a quick look at the data structure

  ```python
  df.head() 
  df.info() # get a description of the data, in particular the total number of rows, and each attributes' type and number of non-null values 

  df["ocean_proximity"].value_counts() # for categorical feature
  df.describe() # get statistical summary of numerical features
  df.hist(bin=50, figsize=(20, 15))

  ```

  * Working with preprocessed attributes is common in machine learning, and it is not necessarily a problem, but you should try to understand how the data is computed. (Is it capped at certain upper and lower bound ? Is the target capped ? Should it be removed ? Or should you collect proper labels ?)
  * Does attributes have very different scales ? 
  * Are the histograms tail heavy ? Do they need to be transformed onto bell-shaped ? 

* Create a test set 

  ```python
  train_set, test_set = train_test_split(hdf, test_size=0.2, random_state=42) # For large dataset (espeically relatvie to the number of attributes) 

  # For smaller dataset, stratefy sampling is important. Be careful of numerical features

  df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1,2,3,4,5])
  split = StratefiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

  for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
  ```


**Discover and visualize the data to gain insights**

* Visualizing data

```python
df = strat_train_set.copy()

df.plot(
    kind="scatter", 
    x="longitude",
    y="latitude",
    alpha=0.4, # setting alpha would allow the figure to show density-related info
    s=df["population"]/100, #radius of each circle represetns the district's population
    label="population",
    figsize=(10, 7),
    c="median_housing_value", # colour represents price
    cmap=plt.get_cmap("jet"), # predefined color map, ranges from blue to red
    colorbar=True
)

# Is there clusters ? Would it be good to use a clustering algo to detect the main cluster, and then add new features that measure the proximity to the cluster center. 
```
* Looking for correlations
  
  If the dataset is not too large, standard correlation coeffecient (Pearson's r) can be easily computed. However, it only measures linear correlations, and it may completely missout on nonlinear relationship. 

  ```python
  corr_matrix = df.corr()
  ```

  ANother way to check for correlation between attribute is to use scatter_matrix. 

  ```python
  attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
  scatter_matrix(df[attributes], figsize=(12, 8))
  # and then maybe zoom in to specfic pair to see if there are toher things that need to be teased out.
  ```
* Experimenting with attribute combinations

  * Data quiks -> maybe clean up
  * Tail-heavy distribution -> transform them
  * Try out various attribute combination -> get the combination and then maybe run correlation again 


**Prepare the data for ml algo**

Instead of just doing the preparation manually, you should write function to do that. 

```python
df = strat_train_set.drop("median_house_value", axis=1)
df_labels = strat_train_set["median_house_value"].copy()
```

* Data cleaning

  * Missing features
    
    total_bedrooms feature has some missing value.

    * Get ride of teh corresponding row
    ```python
    df.dropna(subset=["total_bedrooms"])
    ```
    * Get rid of the whole attribute
    ```python
    df.drop("total_bedrooms", axis=1)
    ```

    * Set the value to some value (zero, the mean, the median, etc.)
    ```python
    median = df["total_bedrooms"].median()
    df["total_bedrooms"].fillna(median, inplace=True)
    ```
    For this option, you should compute the median value on the training set, and use it to fill the missing values in training set, but also don't forget to save the median value that you have comptued. Then use it to replace missing values in test set for evaluation, and also apply it to replace missing value in new data once system goes live. 

    ```python
    # using sklearn implementation

    imputer = SimpleImputer(strategy="median")
    # since median can only be computed on numerical attributes, need to create a copy of the data without text attribute
    df_num = df.drop("ocean_proximity", axis=1)
    imputer.fit(df_num)
    imputer.statistics_
    X = imputer.transform(df_num) # The result is a plain NumPy array
    df_tr = pd.DataFrame(X, columns=df_num.columns)
    ```

* Handling text and categorical attirubtes
  
  Most ml algos prefer to work with numbers, should consider convert the categories from text to numbers.
  ```python
  ordinal_encoder = OrdinalEncoder()
  df_cat_encoded = ordinal_encoder.fit_transform(df_cat)

  ordinal_encoder.categories_ # Get the list of categories 
  ```

  The ordinalEncoder will introduce the oridnal meaning into the categoy, and sometimes it is undesirable. So should consider onehotencoder

  ```python
  cat_encoder = OneHotEncoder()
  df_cat_1hot = cat_encoder.fit_transform(df_cat)
  # The above would transform the feature into a Scipy Sparse matrix. It can be converted into a dense numpyarray be below, but it is less desirable as it is not space optimized. 
  df_cat_1hot.toarray()
  cat_encoder.categories_
  ```
  If the categorical attribute has a large number of categories, then one-hot encoder will result in a large number of input features. You may want to replace it with useful numerical features to related to the categories, or replace each category with a learnable low dimentional vector called embedding. 

* Custom trnasformers

  To create custom transformers, you just need to create a class and implement three menthods: fit() (returning self), transform(), and fit_transform(). you can get the last one for free by simplying adding TransformerMixin as a base class. And if you add BaseEstimator as a base class (without *args and **kargs in the constructor), you will get get_params() and set_params. 

  ```python
  from sklearn.base import BaseEstimator, TransformerMixin

  class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
      self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
      return self
    
    def transform(self, X, y=None):
      rooms_per_household = X[:, rooms_id] / X[:, households_ix]
      population_per_household = X[:, population_ix] / X[:, households_ix]
      if self.add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household,population_per_household,
        bedrooms_per_room]
      else:
        return np.c_[X, rooms_per_household,population_per_household]

  attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
  housing_extra_attrbs = attr_adder.transform(housing.values)
  ```
* Feature scaling

  With few exceptions, ml algos don't performwell when the input numerical attributes have very different scales. Scaling the target values is generally not required. 

  * Min-max scaling (normalization)

    sklearn provides a transformer called MinMaxScaler. It has a feature_range hyperparameter that lets you change the range if you don;t want 0-1 for some reason.  

  * Standardization

    Unlike min-scaling, it does not bound values to a specific range, which may be a problem for some algo (nn often expect an input value ranging form 0 to 1). However, ti is much less affected by outliers. sklearn provides a transformer called StandardScaler for standardization.

* Transformation pipelines

  ```python
  num_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("attribs_adder", CombinedAttributesAdder()),
      ("std_scaler", StandardScaler())
  ])
  housing_num_tr = num_pipeline.fit_transform(housing_num)
  # The pipeline constructor takes a list of name/estimator pairs defining a sequence of steps. All but the last estimator must be transformers (must have a fit_transform()method. ) The name can be anything as long as they are unique and don't contain double underscores "__"

  # The pipeline exploses the same methods as the final estimator. 
  ```

  How about combining the process of both cat and num features steps together ? 

  ```python
  num_attribs = list(housing_num)
  cat_attribs = ["ocean_proximity"]

  full_pipeline = ColumnTransformer([
      ("num", num_pipeline, num_attribs), # tuple containing name, transformer, and a list of names of columns 
      ("cat", OneHotEncoder(), cat_attribs)
  ]) # it applies each transformer to the appropriate columsn and concatenates the outputs along the second axis (the transformers must return the same number of rows). 

  # in the above case the OneHotEncoder returns a sparse matrix, while the num_pipeline retursn a dense matrix. When there is such a mix of sparse and dense matrix, the ColumnTransformer estimates the density of the final matrix (the ratio of non-zero cells) and it returns a sparse matrix if the density is lower than a given threshold (by default sparse_threshold=0.3)


  housing_prepared = full_pipeline.fit_transform(housing)
  ```

  Instead of a trnasformer, you can specify the string "drop" if you want the columsn to be dropped, or specify "pass through" if you want the columns to be left untouched. By default the columsn that were not in the list wuld be dropped, but you can set the remainder hyperparameter to any trnasformer (or to "passthrough") if you want these columsn to be handled differently. 


**Select a model and train it**

* Training and evaluating on the training set

  ```python
  from sklearn.model_selection import cross_val_score
  scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
  tree_rmse_score = np.sqrt(-scores)

  
  ```

Try out many other models from various categories without spending too much time tweaking the hyperparameters first to shortlist a few (2-5) promising models.

Save every model you experiment with, both hyperparameters and trained parameters, as well as the cross-validation socres and perhaps the actual prediction. You can do so with pickle or sklearn.externals.joblib, which is more efficient at serializing large NumPy array:

```python
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")
```

**Fine-tune the model**



**Present your solution**

**Launch, monitor, and maintain your system**