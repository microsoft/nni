# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import responses

def mock_check_status():
    responses.add(
        responses.GET,
        "http://localhost:8080/api/v1/nni/check-status",
        json={"status":"RUNNING","errors":[]},
        status=200
    )

def mock_version():
    responses.add(
        responses.GET,
        "http://localhost:8080/api/v1/nni/version",
        json={'value':1.8},
        status=200
    )

def mock_get_experiment_profile():
    responses.add(
        responses.GET,
        "http://localhost:8080/api/v1/nni/experiment",
        json={"id":"bkfhOdUl","revision":5,"execDuration":10,"logDir":"/home/shinyang/nni-experiments/bkfhOdUl",
        "nextSequenceId":2,"params":{"authorName":"default","experimentName":"example_sklearn-classification",
        "trialConcurrency":1,"maxExecDuration":3600,"maxTrialNum":1,
        "searchSpace":"{\"C\": {\"_type\": \"uniform\", \"_value\": [0.1, 1]}, \
        \"kernel\": {\"_type\": \"choice\", \"_value\": [\"linear\", \"rbf\", \"poly\", \"sigmoid\"]}, \
        \"degree\": {\"_type\": \"choice\", \"_value\": [1, 2, 3, 4]}, \"gamma\": {\"_type\": \"uniform\", \
        \"_value\": [0.01, 0.1]}}", \
        "trainingServicePlatform":"local","tuner":{"builtinTunerName":"TPE","classArgs":{"optimize_mode":"maximize"}, \
        "checkpointDir":"/home/shinyang/nni-experiments/bkfhOdUl/checkpoint"},"versionCheck":"true", \
        "clusterMetaData":[{"key":"codeDir","value":"/home/shinyang/folder/examples/trials/sklearn/classification/."}, \
        {"key":"command","value":"python3 main.py"}]},"startTime":1600326895536,"endTime":1600326910605},
        status=200
    )
 
def mock_update_experiment_profile():
    responses.add(
        responses.PUT, 'http://localhost:8080/api/v1/nni/experiment',
        json={"status":"RUNNING","errors":[]},
        status=200,
        content_type='application/json',
    )

def mock_import_data():
    responses.add(
        responses.POST, 'http://localhost:8080/api/v1/nni/experiment/import-data',
        json={"result":"data"},
        status=201,
        content_type='application/json',
    )

def mock_start_experiment():
    responses.add(
        responses.POST, 'http://localhost:8080/api/v1/nni/experiment',
        json={"status":"RUNNING","errors":[]},
        status=201,
        content_type='application/json',
    )

def mock_get_trial_job_statistics():
    responses.add(
        responses.GET, 'http://localhost:8080/api/v1/nni/job-statistics',
        json=[{"trialJobStatus":"SUCCEEDED","trialJobNumber":1}],
        status=200,
        content_type='application/json',
    )

def mock_set_cluster_metadata():
    responses.add(
        responses.PUT, 'http://localhost:8080/api/v1/nni/experiment/cluster-metadata',
        json=[{"trialJobStatus":"SUCCEEDED","trialJobNumber":1}],
        status=201,
        content_type='application/json',
    )

def mock_list_trial_jobs():
    responses.add(
        responses.GET, 'http://localhost:8080/api/v1/nni/trial-jobs',
        json=[{"id":"GPInz","status":"SUCCEEDED","hyperParameters":["{\"parameter_id\":0, \
        \"parameter_source\":\"algorithm\",\"parameters\":{\"C\":0.8748364659110364, \
        \"kernel\":\"linear\",\"degree\":1,\"gamma\":0.040451413392113666}, \
        \"parameter_index\":0}"],"logPath":"file://localhost:/home/shinyang/nni-experiments/bkfhOdUl/trials/GPInz",
        "startTime":1600326905581,"sequenceId":0,"endTime":1600326906629,
        "finalMetricData":[{"timestamp":1600326906493,"trialJobId":"GPInz","parameterId":"0",
        "type":"FINAL","sequence":0,"data":"\"0.9866666666666667\""}]}],
        status=200,
        content_type='application/json',
    )

def mock_get_trial_job():
    responses.add(
        responses.GET, 'http://localhost:8080/api/v1/nni/trial-jobs/:id',
        json={"id":"GPInz","status":"SUCCEEDED","hyperParameters":["{\"parameter_id\":0, \
        \"parameter_source\":\"algorithm\",\"parameters\":{\"C\":0.8748364659110364, \
        \"kernel\":\"linear\",\"degree\":1,\"gamma\":0.040451413392113666}, \
        \"parameter_index\":0}"],"logPath":"file://localhost:/home/shinyang/nni-experiments/bkfhOdUl/trials/GPInz",
        "startTime":1600326905581,"sequenceId":0,"endTime":1600326906629,
        "finalMetricData":[{"timestamp":1600326906493,"trialJobId":"GPInz","parameterId":"0","type":"FINAL",
        "sequence":0,"data":"\"0.9866666666666667\""}]},
        status=200,
        content_type='application/json',
    )

def mock_add_trial_job():
    responses.add(
        responses.POST, 'http://localhost:8080/api/v1/nni/trial-jobs',
        json=[{"trialJobStatus":"SUCCEEDED","trialJobNumber":1}],
        status=201,
        content_type='application/json',
    )

def mock_cancel_trial_job():
    responses.add(
        responses.DELETE, 'http://localhost:8080/api/v1/nni/trial-jobs/:id',
        json=[{"trialJobStatus":"SUCCEEDED","trialJobNumber":1}],
        status=200,
        content_type='application/json',
    )

def mock_get_metric_data():
    responses.add(
        responses.DELETE, 'http://localhost:8080/api/v1/nni/metric-data/:job_id*?',
        json=[{"timestamp":1600326906486,"trialJobId":"GPInz","parameterId":"0",
        "type":"PERIODICAL","sequence":0,"data":"\"0.9866666666666667\""},
        {"timestamp":1600326906493,"trialJobId":"GPInz","parameterId":"0",
        "type":"FINAL","sequence":0,"data":"\"0.9866666666666667\""}],
        status=200,
        content_type='application/json',
    )

def mock_get_metric_data_by_range():
    responses.add(
        responses.DELETE, 'http://localhost:8080/api/v1/nni/metric-data-range/:min_seq_id/:max_seq_id',
        json=[{"timestamp":1600326906486,"trialJobId":"GPInz","parameterId":"0",
        "type":"PERIODICAL","sequence":0,"data":"\"0.9866666666666667\""},
        {"timestamp":1600326906493,"trialJobId":"GPInz","parameterId":"0",
        "type":"FINAL","sequence":0,"data":"\"0.9866666666666667\""}],
        status=200,
        content_type='application/json',
    )

def mock_get_latest_metric_data():
    responses.add(
        responses.DELETE, 'http://localhost:8080/api/v1/nni/metric-data-latest/',
        json=[{"timestamp":1600326906493,"trialJobId":"GPInz","parameterId":"0",
        "type":"FINAL","sequence":0,"data":"\"0.9866666666666667\""},{"timestamp":1600326906486,
        "trialJobId":"GPInz","parameterId":"0","type":"PERIODICAL",
        "sequence":0,"data":"\"0.9866666666666667\""}],
        status=200,
        content_type='application/json',
    )

def mock_get_trial_log():
    responses.add(
        responses.DELETE, 'http://localhost:8080/api/v1/nni/trial-log/:id/:type',
        json={"status":"RUNNING","errors":[]},
        status=200,
        content_type='application/json',
    )

def mock_export_data():
    responses.add(
        responses.DELETE, 'http://localhost:8080/api/v1/nni/export-data',
        json={"status":"RUNNING","errors":[]},
        status=200,
        content_type='application/json',
    )

def init_response():
    mock_check_status()
    mock_version()
    mock_get_experiment_profile()
    mock_set_cluster_metadata()
    mock_list_trial_jobs()
    mock_get_trial_job()
    mock_add_trial_job()
    mock_cancel_trial_job()
    mock_get_metric_data()
    mock_get_metric_data_by_range()
    mock_get_latest_metric_data()
    mock_get_trial_log()
    mock_export_data()
