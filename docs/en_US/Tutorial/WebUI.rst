WebUI
=====

Experiments managerment
-----------------------

Click the tab ``All experiments`` on the nav bar.

.. image:: ../../img/webui-img/managerExperimentList/experimentListNav.png
   :target: ../../img/webui-img/managerExperimentList/experimentListNav.png
   :alt: ExperimentList nav



* On the ``All experiments`` page, you can see all the experiments on your machine. 

.. image:: ../../img/webui-img/managerExperimentList/expList.png
   :target: ../../img/webui-img/managerExperimentList/expList.png
   :alt: Experiments list



* When you want to see more details about an experiment you could click the trial id, look that:

.. image:: ../../img/webui-img/managerExperimentList/toAnotherExp.png
   :target: ../../img/webui-img/managerExperimentList/toAnotherExp.png
   :alt: See this experiment detail



* If has many experiments on the table, you can use the ``filter`` button.

.. image:: ../../img/webui-img/managerExperimentList/expFilter.png
   :target: ../../img/webui-img/managerExperimentList/expFilter.png
   :alt: filter button



View summary page
-----------------

Click the tab ``Overview``.


* On the overview tab, you can see the experiment information and status and the performance of ``top trials``.


.. image:: ../../img/webui-img/full-oview.png
   :target: ../../img/webui-img/full-oview.png
   :alt: overview



* If you want to see experiment search space and config, please click the right button ``Search space`` and ``Config`` (when you hover on this button).

   1. Search space file:


      .. image:: ../../img/webui-img/searchSpace.png
         :target: ../../img/webui-img/searchSpace.png
         :alt: searchSpace



   2. Config file:


      .. image:: ../../img/webui-img/config.png
         :target: ../../img/webui-img/config.png
         :alt: config



* You can view and download ``nni-manager/dispatcher log files`` on here.


.. image:: ../../img/webui-img/review-log.png
   :target: ../../img/webui-img/review-log.png
   :alt: logfile



* If your experiment has many trials, you can change the refresh interval here.


.. image:: ../../img/webui-img/refresh-interval.png
   :target: ../../img/webui-img/refresh-interval.png
   :alt: refresh




* You can review and download the experiment results(``experiment config``, ``trial message`` and ``intermeidate metrics``) when you click the button ``Experiment summary``.


.. image:: ../../img/webui-img/summary.png
   :target: ../../img/webui-img/summary.png
   :alt: summary



* You can change some experiment configurations such as ``maxExecDuration``, ``maxTrialNum`` and ``trial concurrency`` on here.


.. image:: ../../img/webui-img/edit-experiment-param.png
   :target: ../../img/webui-img/edit-experiment-param.png
   :alt: editExperimentParams



* You can click the icon to see specific error message and ``nni-manager/dispatcher log files`` by clicking ``Learn about`` link.


.. image:: ../../img/webui-img/experimentError.png
   :target: ../../img/webui-img/experimentError.png
   :alt: experimentError




* You can click ``About`` to see the version and report any questions.

View job default metric
-----------------------


* Click the tab ``Default Metric`` to see the point graph of all trials. Hover to see its specific default metric and search space message.


.. image:: ../../img/webui-img/default-metric.png
   :target: ../../img/webui-img/default-metric.png
   :alt: defaultMetricGraph



* Click the switch named ``optimization curve`` to see the experiment's optimization curve.


.. image:: ../../img/webui-img/best-curve.png
   :target: ../../img/webui-img/best-curve.png
   :alt: bestCurveGraph


View hyper parameter
--------------------

Click the tab ``Hyper Parameter`` to see the parallel graph.


* You can ``add/remove`` axes and drag to swap axes on the chart.
* You can select the percentage to see top trials.


.. image:: ../../img/webui-img/hyperPara.png
   :target: ../../img/webui-img/hyperPara.png
   :alt: hyperParameterGraph



View Trial Duration
-------------------

Click the tab ``Trial Duration`` to see the bar graph.


.. image:: ../../img/webui-img/trial_duration.png
   :target: ../../img/webui-img/trial_duration.png
   :alt: trialDurationGraph



View Trial Intermediate Result Graph
------------------------------------

Click the tab ``Intermediate Result`` to see the line graph.


.. image:: ../../img/webui-img/trials_intermeidate.png
   :target: ../../img/webui-img/trials_intermeidate.png
   :alt: trialIntermediateGraph



The trial may have many intermediate results in the training process. In order to see the trend of some trials more clearly, we set a filtering function for the intermediate result graph.

You may find that these trials will get better or worse at an intermediate result. This indicates that it is an important and relevant intermediate result. To take a closer look at the point here, you need to enter its corresponding X-value at #Intermediate. Then input the range of metrics on this intermedia result. In the picture below, we choose the No. 4 intermediate result and set the range of metrics to 0.8-1.


.. image:: ../../img/webui-img/filter-intermediate.png
   :target: ../../img/webui-img/filter-intermediate.png
   :alt: filterIntermediateGraph



View trials status
------------------

Click the tab ``Trials Detail`` to see the status of all trials. Specifically:


* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy, and search space file.


.. image:: ../../img/webui-img/detail-local.png
   :target: ../../img/webui-img/detail-local.png
   :alt: detailLocalImage



* Support searching for a specific trial by its id, status, Trial No. and trial parameters.

1. Trial id: 

.. image:: ../../img/webui-img/detail/searchId.png
   :target: ../../img/webui-img/detail/searchId.png
   :alt: searchTrialId


2. Trial No.: 

.. image:: ../../img/webui-img/detail/searchNo.png
   :target: ../../img/webui-img/detail/searchNo.png
   :alt: searchTrialNo.


3. Trial status:

.. image:: ../../img/webui-img/detail/searchStatus.png
   :target: ../../img/webui-img/detail/searchStatus.png
   :alt: searchStatus

4. Trial parameters:

(1) parameters whose type is choice:

.. image:: ../../img/webui-img/detail/searchParameterChoice.png
   :target: ../../img/webui-img/detail/searchParameterChoice.png
   :alt: searchParameterChoice

(2) parameters whose type is not choice:

.. image:: ../../img/webui-img/detail/searchParameterRange.png
   :target: ../../img/webui-img/detail/searchParameterRange.png
   :alt: searchParameterRange


* The button named ``Add column`` can select which column to show on the table. If you run an experiment whose final result is a dict, you can see other keys in the table. You can choose the column ``Intermediate count`` to watch the trial's progress.


.. image:: ../../img/webui-img/addColumn.png
   :target: ../../img/webui-img/addColumn.png
   :alt: addColumnGraph



* If you want to compare some trials, you can select them and then click ``Compare`` to see the results.


.. image:: ../../img/webui-img/select-trial.png
   :target: ../../img/webui-img/select-trial.png
   :alt: selectTrialGraph


.. image:: ../../img/webui-img/compare.png
   :target: ../../img/webui-img/compare.png
   :alt: compareTrialsGraph


* ``Tensorboard`` please refer `doc <Tensorboard.rst>`__.


* You can use the button named ``Copy as python`` to copy the trial's parameters.


.. image:: ../../img/webui-img/copyParameter.png
   :target: ../../img/webui-img/copyParameter.png
   :alt: copyTrialParameters



* You could see trial logs on the tab of ``Log``. There are three buttons ``View trial log``, ``View trial error`` and ``View trial stdout`` on local mode. If you run on the OpenPAI or Kubeflow platform, you could see trial stdout and nfs log.

1. local mode:

.. image:: ../../img/webui-img/detail/log-local.png
   :target: ../../img/webui-img/detail/log-local.png
   :alt: logOnLocal


2. OpenPAI, Kubeflow and other mode:

.. image:: ../../img/webui-img/detail-pai.png
   :target: ../../img/webui-img/detail-pai.png
   :alt: detailPai


* Intermediate Result Graph: you can see the default metric in this graph by clicking the intermediate button.


.. image:: ../../img/webui-img/intermediate.png
   :target: ../../img/webui-img/intermediate.png
   :alt: intermeidateGraph



* Kill: you can kill a job that status is running.


.. image:: ../../img/webui-img/kill-running.png
   :target: ../../img/webui-img/kill-running.png
   :alt: killTrial



* Customized trial: you can change this trial parameters and then submit it to the experiment. If you want to rerun a failed trial you could submit the same parameters to the experiment.

.. image:: ../../img/webui-img/detail/customizedTrialButton.png
   :target: ../../img/webui-img/detail/customizedTrialButton.png
   :alt: customizedTrialButton



.. image:: ../../img/webui-img/detail/customizedTrial.png
   :target: ../../img/webui-img/detail/customizedTrial.png
   :alt: customizedTrial
