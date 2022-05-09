import React, { lazy, Suspense } from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { getPrefix } from './static/function';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
const Overview = lazy(() => import('./components/experiment/overview/Overview'));
const TrialsDetail = lazy(() => import('./components/experiment/trialdetail/TrialsDetail'));
const ExperimentManagerIndex = lazy(() => import('./components/experimentManagement/ExperimentManagerIndex'));
import '@style/index.css';
import '@style/loading.scss';
import * as serviceWorker from './serviceWorker';

const path = getPrefix();

ReactDOM.render(
    <Router basename={path === undefined ? null : path}>
        <Suspense
            fallback={
                <div className='loading'>
                    <img title='loading-graph' src={(path || '') + '/loading.gif'} />
                </div>
            }
        >
            <Route path='/experiment' component={ExperimentManagerIndex} exact />
            <Switch>
                <App>
                    <Route path='/' component={Overview} exact />
                    <Route path='/oview' component={Overview} />
                    <Route path='/detail' component={TrialsDetail} />
                </App>
            </Switch>
        </Suspense>
    </Router>,

    document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
