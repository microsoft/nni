Feature Engineering with NNI
============================

We are glad to announce the alpha release for Feature Engineering toolkit on top of NNI, it's still in the experiment phase which might evolve based on user feedback. We'd like to invite you to use, feedback and even contribute.

For now, we support the following feature selector:


* `GradientFeatureSelector <./GradientFeatureSelector.rst>`__
* `GBDTSelector <./GBDTSelector.rst>`__

These selectors are suitable for tabular data(which means it doesn't include image, speech and text data).

In addition, those selector only for feature selection. If you want to:
1) generate high-order combined features on nni while doing feature selection;
2) leverage your distributed resources;
you could try this :githublink:`example <examples/feature_engineering/auto-feature-engineering>`.

How to use?
-----------

.. code-block:: python

   from nni.feature_engineering.gradient_selector import FeatureGradientSelector
   # from nni.feature_engineering.gbdt_selector import GBDTSelector

   # load data
   ...
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   # initlize a selector
   fgs = FeatureGradientSelector(...)
   # fit data
   fgs.fit(X_train, y_train)
   # get improtant features
   # will return the index with important feature here.
   print(fgs.get_selected_features(...))

   ...

When using the built-in Selector, you first need to ``import`` a feature selector, and ``initialize`` it. You could call the function ``fit`` in the selector to pass the data to the selector. After that, you could use ``get_seleteced_features`` to get important features. The function parameters in different selectors might be different, so you need to check the docs before using it. 

How to customize?
-----------------

NNI provides *state-of-the-art* feature selector algorithm in the builtin-selector. NNI also supports to build a feature selector by yourself.

If you want to implement a customized feature selector, you need to:


#. Inherit the base FeatureSelector class
#. Implement *fit* and _get_selected*features* function
#. Integrate with sklearn (Optional)

Here is an example:

**1. Inherit the base Featureselector Class**

.. code-block:: python

   from nni.feature_engineering.feature_selector import FeatureSelector

   class CustomizedSelector(FeatureSelector):
       def __init__(self, ...):
       ...

**2. Implement *fit* and _get_selected*features* Function**

.. code-block:: python

   from nni.tuner import Tuner

   from nni.feature_engineering.feature_selector import FeatureSelector

   class CustomizedSelector(FeatureSelector):
       def __init__(self, ...):
       ...

       def fit(self, X, y, **kwargs):
           """
           Fit the training data to FeatureSelector

           Parameters
           ------------
           X : array-like numpy matrix
           The training input samples, which shape is [n_samples, n_features].
           y: array-like numpy matrix
           The target values (class labels in classification, real numbers in regression). Which shape is [n_samples].
           """
           self.X = X
           self.y = y
           ...

       def get_selected_features(self):
           """
           Get important feature

           Returns
           -------
           list :
           Return the index of the important feature.
           """
           ...
           return self.selected_features_

       ...

**3. Integrate with Sklearn**

``sklearn.pipeline.Pipeline`` can connect models in series, such as feature selector, normalization, and classification/regression to form a typical machine learning problem workflow. 
The following step could help us to better integrate with sklearn, which means we could treat the customized feature selector as a module of the pipeline.


#. Inherit the calss *sklearn.base.BaseEstimator*
#. Implement _get\ *params* and _set*params* function in *BaseEstimator*
#. Inherit the class _sklearn.feature\ *selection.base.SelectorMixin*
#. Implement _get\ *support*\ , *transform* and _inverse*transform* Function in *SelectorMixin*

Here is an example:

**1. Inherit the BaseEstimator Class and its Function**

.. code-block:: python

   from sklearn.base import BaseEstimator
   from nni.feature_engineering.feature_selector import FeatureSelector

   class CustomizedSelector(FeatureSelector, BaseEstimator):
       def __init__(self, ...):
       ...

       def get_params(self, ...):
           """
           Get parameters for this estimator.
           """
           params = self.__dict__
           params = {key: val for (key, val) in params.items()
           if not key.endswith('_')}
           return params

       def set_params(self, **params):
           """
           Set the parameters of this estimator.
           """
           for param in params:
           if hasattr(self, param):
           setattr(self, param, params[param])
           return self

**2. Inherit the SelectorMixin Class and its Function**

.. code-block:: python

   from sklearn.base import BaseEstimator
   from sklearn.feature_selection.base import SelectorMixin

   from nni.feature_engineering.feature_selector import FeatureSelector

   class CustomizedSelector(FeatureSelector, BaseEstimator, SelectorMixin):
       def __init__(self, ...):
           ...

       def get_params(self, ...):
           """
           Get parameters for this estimator.
           """
           params = self.__dict__
           params = {key: val for (key, val) in params.items()
           if not key.endswith('_')}
           return params

       def set_params(self, **params):
           """
           Set the parameters of this estimator.
           """
           for param in params:
           if hasattr(self, param):
           setattr(self, param, params[param])
           return self

       def get_support(self, indices=False):
           """
           Get a mask, or integer index, of the features selected.

           Parameters
           ----------
           indices : bool
           Default False. If True, the return value will be an array of integers, rather than a boolean mask.

           Returns
           -------
           list :
           returns support: An index that selects the retained features from a feature vector.
           If indices are False, this is a boolean array of shape [# input features], in which an element is True iff its corresponding feature is selected for retention.
           If indices are True, this is an integer array of shape [# output features] whose values
           are indices into the input feature vector.
           """
           ...
           return mask


       def transform(self, X):
           """Reduce X to the selected features.

           Parameters
           ----------
           X : array
           which shape is [n_samples, n_features]

           Returns
           -------
           X_r : array
           which shape is [n_samples, n_selected_features]
           The input samples with only the selected features.
           """
           ...
           return X_r


       def inverse_transform(self, X):
           """
           Reverse the transformation operation

           Parameters
           ----------
           X : array
           shape is [n_samples, n_selected_features]

           Returns
           -------
           X_r : array
           shape is [n_samples, n_original_features]
           """
           ...
           return X_r

After integrating with Sklearn, we could use the feature selector as follows:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression

   # load data
   ...
   X_train, y_train = ...

   # build a ppipeline
   pipeline = make_pipeline(XXXSelector(...), LogisticRegression())
   pipeline = make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
   pipeline.fit(X_train, y_train)

   # score
   print("Pipeline Score: ", pipeline.score(X_train, y_train))

Benchmark
---------

``Baseline`` means without any feature selection, we directly pass the data to LogisticRegression. For this benchmark, we only use 10% data from the train as test data. For the GradientFeatureSelector, we only take the top20 features. The metric is the mean accuracy on the given test data and labels.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Dataset
     - All Features + LR (acc, time, memory)
     - GradientFeatureSelector + LR (acc, time, memory)
     - TreeBasedClassifier + LR (acc, time, memory)
     - #Train
     - #Feature
   * - colon-cancer
     - 0.7547, 890ms, 348MiB
     - 0.7368, 363ms, 286MiB
     - 0.7223, 171ms, 1171 MiB
     - 62
     - 2,000
   * - gisette
     - 0.9725, 215ms, 584MiB
     - 0.89416, 446ms, 397MiB
     - 0.9792, 911ms, 234MiB
     - 6,000
     - 5,000
   * - avazu
     - 0.8834, N/A, N/A
     - N/A, N/A, N/A
     - N/A, N/A, N/A
     - 40,428,967
     - 1,000,000
   * - rcv1
     - 0.9644, 557ms, 241MiB
     - 0.7333, 401ms, 281MiB
     - 0.9615, 752ms, 284MiB
     - 20,242
     - 47,236
   * - news20.binary
     - 0.9208, 707ms, 361MiB
     - 0.6870, 565ms, 371MiB
     - 0.9070, 904ms, 364MiB
     - 19,996
     - 1,355,191
   * - real-sim
     - 0.9681, 433ms, 274MiB
     - 0.7969, 251ms, 274MiB
     - 0.9591, 643ms, 367MiB
     - 72,309
     - 20,958


The dataset of benchmark could be download in `here <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/>`__

The code could be refenrence ``/examples/feature_engineering/gradient_feature_selector/benchmark_test.py``.

Reference and Feedback
----------------------


* To `report a bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__ for this feature in GitHub;
* To `file a feature or improvement request <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__ for this feature in GitHub;
* To know more about :githublink:`Neural Architecture Search with NNI <docs/en_US/NAS/Overview.rst>`\ ;
* To know more about :githublink:`Model Compression with NNI <docs/en_US/Compression/Overview.rst>`\ ;
* To know more about :githublink:`Hyperparameter Tuning with NNI <docs/en_US/Tuner/BuiltinTuner.rst>`\ ;
