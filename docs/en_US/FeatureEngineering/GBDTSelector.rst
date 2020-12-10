GBDTSelector
------------

GBDTSelector is based on `LightGBM <https://github.com/microsoft/LightGBM>`__\ , which is a gradient boosting framework that uses tree-based learning algorithms.

When passing the data into the GBDT model, the model will construct the boosting tree. And the feature importance comes from the score in construction, which indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model.

We could use this method as a strong baseline in Feature Selector, especially when using the GBDT model as a classifier or regressor.

For now, we support the ``importance_type`` is ``split`` and ``gain``. But we will support customized ``importance_type`` in the future, which means the user could define how to calculate the ``feature score`` by themselves.

Usage
^^^^^

First you need to install dependency:

.. code-block:: bash

   pip install lightgbm

Then

.. code-block:: python

   from nni.feature_engineering.gbdt_selector import GBDTSelector

   # load data
   ...
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   # initlize a selector
   fgs = GBDTSelector()
   # fit data
   fgs.fit(X_train, y_train, ...)
   # get improtant features
   # will return the index with important feature here.
   print(fgs.get_selected_features(10))

   ...

And you could reference the examples in ``/examples/feature_engineering/gbdt_selector/``\ , too.

**Requirement of ``fit`` FuncArgs**


* 
  **X** (array-like, require) - The training input samples which shape = [n_samples, n_features]

* 
  **y** (array-like, require) - The target values (class labels in classification, real numbers in regression) which shape = [n_samples].

* 
  **lgb_params** (dict, require) - The parameters for lightgbm model. The detail you could reference `here <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`__

* 
  **eval_ratio** (float, require) - The ratio of data size. It's used for split the eval data and train data from self.X.

* 
  **early_stopping_rounds** (int, require) - The early stopping setting in lightgbm. The detail you could reference `here <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`__.

* 
  **importance_type** (str, require) - could be 'split' or 'gain'. The 'split' means ' result contains numbers of times the feature is used in a model' and the 'gain' means 'result contains total gains of splits which use the feature'. The detail you could reference in `here <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance>`__.

* 
  **num_boost_round** (int, require) - number of boost round. The detail you could reference `here <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train>`__.

**Requirement of ``get_selected_features`` FuncArgs**


* **topk** (int, require) - the topK impotance features you want to selected.
