.. role:: raw-html(raw)
   :format: html

搜索空间
============

概述
--------

在 NNI 中，Tuner 会根据搜索空间来取样生成参数和网络架构。搜索空间通过 JSON 文件来定义。

要定义搜索空间，需要定义变量名称、采样策略的类型及其参数。


* 搜索空间示例如下：

.. code-block:: yaml

   {
       "dropout_rate": {"_type": "uniform", "_value": [0.1, 0.5]},
       "conv_size": {"_type": "choice", "_value": [2, 3, 5, 7]},
       "hidden_size": {"_type": "choice", "_value": [124, 512, 1024]},
       "batch_size": {"_type": "choice", "_value": [50, 250, 500]},
       "learning_rate": {"_type": "uniform", "_value": [0.0001, 0.1]}
   }

将第一行作为示例。 ``dropout_rate`` 定义了一个变量，先验分布为均匀分布，范围从 ``0.1`` 到 ``0.5``。

注意：搜索空间中可用的取样策略取决于要使用的 Tuner 。 此处列出了内置 Tuner 所支持的类型。 对于自定义的 Tuner，不必遵循这些约定，可使用任何的类型。

类型
-----

所有采样策略和参数如下：


* 
  ``{"_type": "choice", "_value": options}``


  * 变量值为 options 中之一。 这里的 ``options`` 应该是字符串或数值的列表。 可将任意对象（如子数组，数字与字符串的混合值或者空值）存入此数组中，但可能会产生不可预料的行为。
  * ``options`` 也可以是嵌套的子搜索空间。此子搜索空间仅在相应的元素选中后才起作用。 该子搜索空间中的变量可看作是条件变量。 :githublink:`嵌套搜索空间定义的简单示例 <examples/trials/mnist-nested-search-space/search_space.json>`。 如果选项列表中的元素是 dict，则它是一个子搜索空间，对于内置的 Tuner，必须在此 dict 中添加键 ``_name``，这有助于标识选中的元素。 相应的，这是从 NNI 中获得的嵌套搜索空间定义的 :githublink:`示例 <examples/trials/mnist-nested-search-space/sample.json>` 。 参见下表了解支持嵌套搜索空间的 Tuner 。

* 
  ``{"_type": "randint", "_value": [lower, upper]}``


  * 从 ``lower`` (包含) 到 ``upper`` (不包含) 中选择一个随机整数。
  * 注意：不同 Tuner 可能对 ``randint`` 有不同的实现。 一些 Tuner（例如，TPE，GridSearch）将从低到高无序选择，
    而其它一些（例如，SMAC）则有顺序。 如果希望所有 Tuner 都有序，
    可使用 ``quniform`` 并设置 ``q=1``。

* 
  ``{"_type": "uniform", "_value": [low, high]}``


  * 变量值在 low 和 high 之间均匀采样。
  * 当优化时，此变量值会在两侧区间内。

* 
  ``{"_type": "quniform", "_value": [low, high, q]}``


  * 变量值为 ``clip(round(uniform(low, high) / q) * q, low, high)``，clip 操作用于约束生成值的边界。 例如，``_value`` 为 [0, 10, 2.5]，可取的值为 [0, 2.5, 5.0, 7.5, 10.0]; ``_value`` 为 [2, 10, 5]，可取的值为 [2, 5, 10]。
  * 适用于离散，同时反映了某种"平滑"的数值，但上下限都有限制。 如果需要从范围 [low, high] 中均匀选择整数，可以如下定义 ``_value``：``[low, high, 1]``。

* 
  ``{"_type": "loguniform", "_value": [low, high]}``


  * 变量值在范围 [low, high] 中是 loguniform 分布，如 exp(uniform(log(low), log(high)))，因此返回值是对数均匀分布的。
  * 当优化时，此变量必须是正数。

* 
  ``{"_type": "qloguniform", "_value": [low, high, q]}``


  * 变量值为 ``clip(round(loguniform(low, high) / q) * q, low, high)``，clip 操作用于约束生成值的边界。
  * 适用于值是“平滑”的离散变量，但上下限均有限制。

* 
  ``{"_type": "normal", "_value": [mu, sigma]}``


  * 变量值为实数，且为正态分布，均值为 mu，标准方差为 sigma。 优化时，此变量不受约束。

* 
  ``{"_type": "qnormal", "_value": [mu, sigma, q]}``


  * 变量的值由 ``round(normal(mu, sigma) / q) * q`` 确定。
  * 适用于在 mu 周围的离散变量，且没有上下限限制。

* 
  ``{"_type": "lognormal", "_value": [mu, sigma]}``


  * 变量值为 ``exp(normal(mu, sigma))`` 分布，范围值是对数的正态分布。 当优化时，此变量必须是正数。

* 
  ``{"_type": "qlognormal", "_value": [mu, sigma, q]}``


  * 变量的值由 ``round(exp(normal(mu, sigma)) / q) * q`` 确定。
  * 适用于值是“平滑”的离散变量，但某一边有界。

每种 Tuner 支持的搜索空间类型
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 
     - choice
     - choice(nested)
     - randint
     - uniform
     - quniform
     - loguniform
     - qloguniform
     - normal
     - qnormal
     - lognormal
     - qlognormal
   * - TPE Tuner
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
   * - Random Search Tuner
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
   * - Anneal Tuner
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
   * - Evolution Tuner
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
   * - SMAC Tuner
     - :raw-html:`&#10003;`
     - 
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - 
     - 
     - 
     - 
     - 
   * - Batch Tuner
     - :raw-html:`&#10003;`
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Grid Search Tuner
     - :raw-html:`&#10003;`
     - 
     - :raw-html:`&#10003;`
     - 
     - :raw-html:`&#10003;`
     - 
     - 
     - 
     - 
     - 
     - 
   * - Hyperband Advisor
     - :raw-html:`&#10003;`
     - 
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
   * - Metis Tuner
     - :raw-html:`&#10003;`
     - 
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - 
     - 
     - 
     - 
     - 
     - 
   * - GP Tuner
     - :raw-html:`&#10003;`
     - 
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - :raw-html:`&#10003;`
     - 
     - 
     - 
     - 


已知的局限：


* 
  GP Tuner 和 Metis Tuner 的搜索空间只支持 **数值**，（**choice** 类型在其它 Tuner 中可以使用非数值， 如：字符串等）。 GP Tuner 和 Metis Tuner 都使用了高斯过程的回归（Gaussian Process Regressor, GPR）。 GPR 基于计算不同点距离的和函数来进行预测，其无法计算非数值值的距离。

* 
  请注意，对于嵌套搜索空间：


  * 只有 随机搜索/TPE/Anneal/Evolution Tuner/Grid Search 支持嵌套搜索空间
