import React, { lazy, Suspense } from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { getPrefix } from './static/function';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
const Overview = lazy(() => import('./components/Overview'));
const TrialsDetail = lazy(() => import('./components/TrialsDetail'));
const Experiment = lazy(() => import('./components/managementExp/ExperimentManager'));
import './index.css';
import './static/style/loading.scss';
import * as serviceWorker from './serviceWorker';

const path = getPrefix();

ReactDOM.render(
    <Router basename={path === undefined ? null : path}>
        <Suspense
            fallback={
                <div className='loading'>
                    <img src={require('./static/img/loading.gif')} />
                </div>
            }
        >
            <Route path='/experiment' component={Experiment} exact />
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
