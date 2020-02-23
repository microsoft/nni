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
app.get('/hey', (req, res) => res.send('ho!'));
app.get('/refresh', (req, res) => {
  const graph = fs.readFileSync(path.join(logdir, 'graph.json'), 'utf8');
  const log = fs.readFileSync(path.join(logdir, 'log.json'), 'utf8');
  res.send({
    'graph': JSON.parse(graph),
    'log': JSON.parse(log)
  });
});

app.listen(8080, () => {
  console.log(`NNI NAS board is running on port ${port}, logdir is ${logdir}.`);
});
