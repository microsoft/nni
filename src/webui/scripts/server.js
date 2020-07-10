const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const argv = require('minimist')(process.argv.slice(2));
const port = argv.port || 12138;
const expAlias = argv.experiment || process.env.EXPERIMENT || 'mnist-tfv1-running';
const expData = JSON.parse(fs.readFileSync(path.join(__dirname, `../mock/${expAlias}.json`)).toString());

app.get('/api/v1/nni/version', (req, res) => {
    res.send('v999.0');
});
app.get('/api/v1/nni/check-status', (req, res) => {
    res.send(expData.checkStatus);
});
app.get('/api/v1/nni/experiment', (req, res) => {
    res.send(expData.experiment);
});
app.get('/api/v1/nni/metric-data', (req, res) => {
    res.send(expData.metricData);
});
app.get('/api/v1/nni/trial-jobs', (req, res) => {
    res.send(expData.trialJobs);
});

app.listen(port, '0.0.0.0', () => {
    console.log(`Listening on port ${port}, serving data: ${expAlias}`);
});
