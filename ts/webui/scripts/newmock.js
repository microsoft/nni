const path = require('path');
const process = require('process');
const fs = require('fs');
const argv = require('minimist')(process.argv.slice(2), {
    string: ['server', 'target'],
    alias: { s: 'server', t: 'target' }
});
const axios = require('axios');
const serverAddr = argv.server;
const targetFileName = argv.target;
if (!serverAddr) {
    throw new Error('Server address not set!');
}
if (!targetFileName) {
    throw new Error('Target file name not set!');
}

function maskSensitive(obj) {
    if (Array.isArray(obj)) {
        return obj.map(maskSensitive);
    } else if (typeof obj === 'object') {
        const ret = {};
        for (const key in obj) {
            ret[maskSensitive(key)] = maskSensitive(obj[key]);
        }
        return ret;
    } else if (typeof obj === 'string') {
        const homeDir = process.env.HOME;
        if (homeDir && obj.includes(homeDir)) {
            obj = obj.replace(homeDir, '/***');
        }
        return obj;
    } else {
        return obj;
    }
}

axios.all([
    axios.get(`${serverAddr}/api/v1/nni/check-status`),
    axios.get(`${serverAddr}/api/v1/nni/experiment`),
    axios.get(`${serverAddr}/api/v1/nni/metric-data`),
    axios.get(`${serverAddr}/api/v1/nni/trial-jobs`)
]).then(axios.spread((checkStatus, experiment, metricData, trialJobs) => {
    const data = JSON.stringify(maskSensitive({
        checkStatus: checkStatus.data,
        experiment: experiment.data,
        metricData: metricData.data,
        trialJobs: trialJobs.data
    }), null, 2);
    fs.writeFileSync(path.join('mock', `${targetFileName}.json`), data);
})).catch(error => {
    console.log(error);
});
