GradientFeatureSelector
-----------------------

The algorithm in GradientFeatureSelector comes from `"Feature Gradients: Scalable Feature Selection via Discrete Relaxation" <https://arxiv.org/pdf/1908.10382.pdf>`__.

GradientFeatureSelector, a gradient-based search algorithm
for feature selection. 

1) This approach extends a recent result on the estimation of
learnability in the sublinear data regime by showing that the calculation can be performed iteratively (i.e., in mini-batches) and in **linear time and space** with respect to both the number of features D and the sample size N. 

2) This, along with a discrete-to-continuous relaxation of the search domain, allows for an **efficient, gradient-based** search algorithm among feature subsets for very **large datasets**.

3) Crucially, this algorithm is capable of finding **higher-order correlations** between features and targets for both the N > D and N < D regimes, as opposed to approaches that do not consider such interactions and/or only consider one regime.

Usage
^^^^^

.. code-block:: python

   from nni.feature_engineering.gradient_selector import FeatureGradientSelector

   # load data
   ...
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   # initlize a selector
   fgs = FeatureGradientSelector(n_features=10)
   # fit data
   fgs.fit(X_train, y_train)
   # get improtant features
   # will return the index with important feature here.
   print(fgs.get_selected_features())

   ...

And you could reference the examples in ``/examples/feature_engineering/gradient_feature_selector/``\ , too.

**Parameters of class FeatureGradientSelector constructor**


* 
  **order** (int, optional, default = 4) - What order of interactions to include. Higher orders may be more accurate but increase the run time. 12 is the maximum allowed order.

* 
  **penatly** (int, optional, default = 1) - Constant that multiplies the regularization term.

* 
  **n_features** (int, optional, default = None) - If None, will automatically choose number of features based on search. Otherwise, the number of top features to select.

* 
  **max_features** (int, optional, default = None) - If not None, will use the 'elbow method' to determine the number of features with max_features as the upper limit.

* 
  **learning_rate** (float, optional, default = 1e-1) - learning rate

* 
  **init** (*zero, on, off, onhigh, offhigh, or sklearn, optional, default = zero*\ ) - How to initialize the vector of scores. 'zero' is the default.

* 
  **n_epochs** (int, optional, default = 1) - number of epochs to run

* 
  **shuffle** (bool, optional, default = True) - Shuffle "rows" prior to an epoch.

* 
  **batch_size** (int, optional, default = 1000) - Nnumber of "rows" to process at a time.

* 
  **target_batch_size** (int, optional, default = 1000) - Number of "rows" to accumulate gradients over. Useful when many rows will not fit into memory but are needed for accurate estimation.

* 
  **classification** (bool, optional, default = True) - If True, problem is classification, else regression.

* 
  **ordinal** (bool, optional, default = True) - If True, problem is ordinal classification. Requires classification to be True.

* 
  **balanced** (bool, optional, default = True) - If true, each class is weighted equally in optimization, otherwise weighted is done via support of each class. Requires classification to be True.

* 
  **prerocess** (str, optional, default = 'zscore') - 'zscore' which refers to centering and normalizing data to unit variance or 'center' which only centers the data to 0 mean.

* 
  **soft_grouping** (bool, optional, default = True) - If True, groups represent features that come from the same source. Used to encourage sparsity of groups and features within groups.

* 
  **verbose** (int, optional, default = 0) - Controls the verbosity when fitting. Set to 0 for no printing 1 or higher for printing every verbose number of gradient steps.

* 
  **device** (str, optional, default = 'cpu') - 'cpu' to run on CPU and 'cuda' to run on GPU. Runs much faster on GPU

**Requirement of ``fit`` FuncArgs**


* 
  **X** (array-like, require) - The training input samples which shape = [n_samples, n_features]

* 
  **y** (array-like, require) - The target values (class labels in classification, real numbers in regression) which shape = [n_samples].

* 
  **groups** (array-like, optional, default = None) - Groups of columns that must be selected as a unit. e.g. [0, 0, 1, 2] specifies the first two columns are part of a group. Which shape is [n_features].

**Requirement of ``get_selected_features`` FuncArgs**

 For now, the ``get_selected_features`` function has no parameters.
