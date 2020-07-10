import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { Router, Route, browserHistory, IndexRedirect } from 'react-router';
import Overview from './components/Overview';
import TrialsDetail from './components/TrialsDetail';
import './index.css';
import * as fs from 'fs';
import * as serviceWorker from './serviceWorker';
import { makeServer } from './server';
import * as mnistTfv1FullData from './mock/mnist-tfv1-running.json';
import * as dictMetrics100 from './mock/dict-metrics-100.json';

if (process.env.NODE_ENV === 'development') {
  const data = (dictMetrics100 as any).default;
  makeServer(data, 'development');
}

ReactDOM.render(
  (
    <Router history={browserHistory}>
      <Route path="/" component={App}>
        <IndexRedirect to="/oview" />
        <Route path="/oview" component={Overview} />
        <Route path="/detail" component={TrialsDetail} />
        {/* test branch */}
      </Route>
    </Router>

  ),
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
