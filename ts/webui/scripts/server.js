const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const argv = require('minimist')(process.argv.slice(2));
const port = argv.port || 12138;
const expAlias = argv.experiment || process.env.EXPERIMENT || 'mnist-tfv1-running';
// Specify multiple alias to transfer from one to another automatically
const splittedAlias = expAlias.split(',');
let expData = undefined;

function loadExperimentWithAlias(aliasId) {
    const alias = splittedAlias[aliasId];
    let expFile = path.join(__dirname, `../mock/${alias}`);
    if (!fs.existsSync(expFile)) {
        expFile += '.json';
    }
    if (!fs.existsSync(expFile)) {
        throw new Error(`Experiment file '${expFile}' not found. Please recheck.`);
    }
    console.log(`Loading experiment file: '${expFile}'.`);
    expData = JSON.parse(fs.readFileSync(expFile).toString());
    if (splittedAlias.length <= 1)
        return;
    // sleep longer on first one
    setTimeout(() => loadExperimentWithAlias((aliasId + 1) % splittedAlias.length),
        expData === undefined ? 40000 : 20000);
}

loadExperimentWithAlias(0);

app.get('/api/v1/nni/version', (req, res) => {
    res.send('v999.0');
});
app.get('/api/v1/nni/check-status', (req, res) => {
    res.send(expData.checkStatus);
});
app.get('/api/v1/nni/experiment', (req, res) => {
    res.send(expData.experiment);
});
app.get('/api/v1/nni/job-statistics', (req, res) => {
    const counter = {};
    for (const t of expData.trialJobs) {
        counter[t.status] = (counter[t.status] || 0) + 1;
    }
    res.send(Object.keys(counter).map((k) => {
        return {
            trialJobStatus: k,
            trialJobNumber: counter[k]
        };
    }));
});

app.get('/api/v1/nni/metric-data', (req, res) => {
    res.send(expData.metricData);
});
app.get('/api/v1/nni/metric-data/:job_id', (req, res) => {
    const metricData = expData.metricData.filter((item) => item.trialJobId === req.params.job_id);
    res.send(metricData);
});
app.get('/api/v1/nni/metric-data-range/:min_seq_id/:max_seq_id', (req, res) => {
    const minSeqId = Number(req.params.min_seq_id);
    const maxSeqId = Number(req.params.max_seq_id);
    const targetTrials = expData.trialJobs.filter(trial => (
        // Copied from nnimanager.ts
        trial.sequenceId !== undefined && minSeqId <= trial.sequenceId && trial.sequenceId <= maxSeqId
    ));
    const targetTrialIds = new Set(targetTrials.map(trial => trial.id));
    res.send(expData.metricData.filter(metric => targetTrialIds.has(metric.trialJobId)));
});
app.get('/api/v1/nni/metric-data-latest', (req, res) => {
    const finals = [];
    const latestIntermediates = new Map();
    for (const metric of expData.metricData) {
        if (metric.type !== 'PERIODICAL') {
            finals.push(metric);
        } else {
            const old = latestIntermediates.get(metric.trialJobId);
            if (old === undefined || old.sequence <= metric.sequence) {
                latestIntermediates.set(metric.trialJobId, metric);
            }
        }
    }
    res.send(finals.concat(Array.from(latestIntermediates.values())));
});

app.get('/api/v1/nni/trial-jobs', (req, res) => {
    res.send(expData.trialJobs);
});
app.get('/api/v1/nni/trial-jobs/:id', (req, res) => {
    for (const t of expData.trialJobs) {
        if (t.id === req.params.id) {
            res.send(t);
            break;
        }
    }
    res.sendStatus(404);
});
// TODO: implement put, post, delete methods

app.listen(port, '0.0.0.0', () => {
    console.log(`Listening on port ${port}, serving data: ${expAlias}`);
});
