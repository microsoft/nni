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



* When you want to see more details about an experiments you could click the row or trial id, look that:

.. image:: ../../img/webui-img/managerExperimentList/toAnotherExp.png
   :target: ../../img/webui-img/managerExperimentList/toAnotherExp.png
   :alt: See this experiment detail



* If has many experiments on the table, you can use the ``filter`` button.

.. image:: ../../img/webui-img/managerExperimentList/expFilter.png
   :target: ../../img/webui-img/managerExperimentList/expFilter.png
   :alt: filter button



View summary page
-----------------

Click the tab "Overview".


* On the overview tab, you can see the experiment information and status and the performance of top trials.


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



* You can view and download nni-manager/dispatcher log files on here.


.. image:: ../../img/webui-img/review-log.png
   :target: ../../img/webui-img/review-log.png
   :alt: logfile



* If your experiment has many trials, you can change the refresh interval here.


.. image:: ../../img/webui-img/refresh-interval.png
   :target: ../../img/webui-img/refresh-interval.png
   :alt: 



* You can review and download the experiment results(experiment config, trial message and intermeidate metric) when you click the button named ``Experiment summary``.


.. image:: ../../img/webui-img/summary.png
   :target: ../../img/webui-img/summary.png
   :alt: 



* You can change some experiment configurations such as maxExecDuration, maxTrialNum and trial concurrency on here.


.. image:: ../../img/webui-img/edit-experiment-param.png
   :target: ../../img/webui-img/edit-experiment-param.png
   :alt: 



* You can click the exclamation point in the error box to see a log message if the experiment's status is an error.


.. image:: ../../img/webui-img/log-error.png
   :target: ../../img/webui-img/log-error.png
   :alt: 


.. image:: ../../img/webui-img/review-log.png
   :target: ../../img/webui-img/review-log.png
   :alt: 



* You can click "About" to see the version and report any questions.

View job default metric
-----------------------


* Click the tab "Default Metric" to see the point graph of all trials. Hover to see its specific default metric and search space message.


.. image:: ../../img/webui-img/default-metric.png
   :target: ../../img/webui-img/default-metric.png
   :alt: 



* Click the switch named "optimization curve" to see the experiment's optimization curve.


.. image:: ../../img/webui-img/best-curve.png
   :target: ../../img/webui-img/best-curve.png
   :alt: 


View hyper parameter
--------------------

Click the tab "Hyper Parameter" to see the parallel graph.


* You can add/remove axes and drag to swap axes on the chart.
* You can select the percentage to see top trials.


.. image:: ../../img/webui-img/hyperPara.png
   :target: ../../img/webui-img/hyperPara.png
   :alt: 


View Trial Duration
-------------------

Click the tab "Trial Duration" to see the bar graph.


.. image:: ../../img/webui-img/trial_duration.png
   :target: ../../img/webui-img/trial_duration.png
   :alt: 


View Trial Intermediate Result Graph
------------------------------------

Click the tab "Intermediate Result" to see the line graph.


.. image:: ../../img/webui-img/trials_intermeidate.png
   :target: ../../img/webui-img/trials_intermeidate.png
   :alt: 


The trial may have many intermediate results in the training process. In order to see the trend of some trials more clearly, we set a filtering function for the intermediate result graph.

You may find that these trials will get better or worse at an intermediate result. This indicates that it is an important and relevant intermediate result. To take a closer look at the point here, you need to enter its corresponding X-value at #Intermediate. Then input the range of metrics on this intermedia result. In the picture below, we choose the No. 4 intermediate result and set the range of metrics to 0.8-1.


.. image:: ../../img/webui-img/filter-intermediate.png
   :target: ../../img/webui-img/filter-intermediate.png
   :alt: 


View trials status
------------------

Click the tab "Trials Detail" to see the status of all trials. Specifically:


* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy, and search space file.


.. image:: ../../img/webui-img/detail-local.png
   :target: ../../img/webui-img/detail-local.png
   :alt: 



* The button named "Add column" can select which column to show on the table. If you run an experiment whose final result is a dict, you can see other keys in the table. You can choose the column "Intermediate count" to watch the trial's progress.


.. image:: ../../img/webui-img/addColumn.png
   :target: ../../img/webui-img/addColumn.png
   :alt: 



* If you want to compare some trials, you can select them and then click "Compare" to see the results.


.. image:: ../../img/webui-img/select-trial.png
   :target: ../../img/webui-img/select-trial.png
   :alt: 


.. image:: ../../img/webui-img/compare.png
   :target: ../../img/webui-img/compare.png
   :alt: 



* Support to search for a specific trial by it's id, status, Trial No. and parameters.


.. image:: ../../img/webui-img/search-trial.png
   :target: ../../img/webui-img/search-trial.png
   :alt: 



* You can use the button named "Copy as python" to copy the trial's parameters.


.. image:: ../../img/webui-img/copyParameter.png
   :target: ../../img/webui-img/copyParameter.png
   :alt: 



* If you run on the OpenPAI or Kubeflow platform, you can also see the nfs log.


.. image:: ../../img/webui-img/detail-pai.png
   :target: ../../img/webui-img/detail-pai.png
   :alt: 



* Intermediate Result Graph: you can see the default metric in this graph by clicking the intermediate button.


.. image:: ../../img/webui-img/intermediate.png
   :target: ../../img/webui-img/intermediate.png
   :alt: 



* Kill: you can kill a job that status is running.


.. image:: ../../img/webui-img/kill-running.png
   :target: ../../img/webui-img/kill-running.png
   :alt: 

