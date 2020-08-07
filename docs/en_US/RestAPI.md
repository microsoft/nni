# RESTful API

## Experiment settings

* Request
  + Method: **Get**
  + URL: `/api/v1/nni/experiment`
* Response
  + Body

    ``` json
    {
      "id":"",
      "revision":188,
      "execDuration":1751,
      "logDir":"",
      "nextSequenceId":11,
      "params":{
        "authorName":"default",
        "experimentName":"",
        "trialConcurrency":1,
        "maxExecDuration":3600,
        "maxTrialNum":10,
        "searchSpace":"{
          \"batch_size\": {\"_type\": \"choice\", \"_value\": [16, 32, 64, 128]},
          \"hidden_size\": {\"_type\": \"choice\", \"_value\": [128, 256, 512, 1024]},
          \"lr\": {\"_type\": \"choice\", \"_value\": [0.0001, 0.001, 0.01, 0.1]},
          \"momentum\": {\"_type\": \"uniform\", \"_value\": [0, 1]}}",
        "trainingServicePlatform":"local",
        "tuner":{
          "builtinTunerName":"TPE",
          "classArgs":{
            "optimize_mode":"maximize"
          },
          "checkpointDir":""
        },
        "versionCheck":true,
        "clusterMetaData":[
          {
            "key":"codeDir",
            "value":""
          },
          {
            "key":"command",
            "value":"python3 mnist.py"
          }
          ]},
      "startTime":1596172409937,
      "endTime":1596174324943
    }
    ```

## Trial details

* Request
  + Method: **Get**
  + URL: `/api/v1/nni/trial-jobs`
* Response
  + Body

    ``` json
    [{
      "id":"eZ5ci",
      "status":"SUCCEEDED",
      "hyperParameters":["{
        \"parameter_id\":0,
        \"parameter_source\":\"algorithm\",
        \"parameters\":{
          \"batch_size\":16,
          \"hidden_size\":1024,
          \"lr\":0.0001,
          \"momentum\":0.2765672596744012
        },
        \"parameter_index\":0}"],
      "logPath":"",
      "startTime":1596172419971,
      "sequenceId":0,
      "endTime":1596172634221,
      "finalMetricData":[{
        "timestamp":1596172633754,
        "trialJobId":"eZ5ci",
        "parameterId":"0",
        "type":"FINAL",
        "sequence":0,
        "data":"\"{\\\"default\\\": 93.64, \\\"other_metric\\\": 2.0}\""
      }]
    },
    ]
    ```

    Note that if using non-dict metric training process, `"data"` in `finalMetricData` should be `"data":"\"93.64\""`.

## All trial results

* Request
  + Method: **Get**
  + URL: `/api/v1/nni/metric-data`
* Response
  + Body

    ``` json
    [
      {
        "timestamp":1596172450331,
        "trialJobId":"eZ5ci",
        "parameterId":"0",
        "type":"PERIODICAL",
        "sequence":0,
        "data":"\"{\\\"default\\\": 70.83, \\\"key\\\": 1.0}\""
      },
      {
        "timestamp":1596172633754,
        "trialJobId":"eZ5ci",
        "parameterId":"0",
        "type":"FINAL",
        "sequence":0,
        "data":"\"{\\\"default\\\": 93.64, \\\"other_metric\\\": 2.0}\""
      },
    ]
    ```

  By default, the returned results are sorted in time order. You can use `/api/v1/nni/metric-data-latest` to place all final results before intermediate results.

## Check experiment status

* Request
  + Method: **Get**
  + URL: `/api/v1/nni/check-status`
* Response
  + Body

    ``` json
    {
      "status":"DONE",
      "errors":[]
    }
    ```

## Export experiments results

* Request
  + Method: **Get**
  + URL: `/api/v1/nni/check-status`
* Response
  + Body

    ``` json
    [
      {
        "parameter":{
          "LSTM_dropout_rate":0.5588629039375548,
          "batch_size":32,
          "learning_rate":0.005984331286321558,
          "embedding_drop":0.5303838118643356,
          "passage_drop":0.37427564947505015,
          "query_drop":0.8232843672322491,
          "decay":0.9879453728036693,
          "loss_weight":0.5396219835360014
        },
        "value":"0.6473530891836636",
        "id":"VRiyI"
      }
    ]
    ```

