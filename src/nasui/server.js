const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const argv = require('minimist')(process.argv.slice(2));
const port = argv.port || 8080;
const logdir = argv.logdir || './mockdata';

app.use(express.static(path.join(__dirname, 'build')));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});
app.get('/refresh', (req, res) => {
  const graph = fs.readFileSync(path.join(logdir, 'graph.json'), 'utf8');
  const log = fs.readFileSync(path.join(logdir, 'log'), 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map(JSON.parse);
  res.send({
    'graph': JSON.parse(graph),
    'log': log,
  });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`NNI NAS board is running on port ${port}, logdir is ${logdir}.`);
});
