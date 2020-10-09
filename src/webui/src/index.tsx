import React, { lazy, Suspense } from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { BrowserRouter as Router, Route, Switch, Redirect } from 'react-router-dom';
const Overview = lazy(() => import('./components/Overview'));
const TrialsDetail = lazy(() => import('./components/TrialsDetail'));
import './index.css';
import * as serviceWorker from './serviceWorker';

ReactDOM.render(
    <Router>
        <App>
            <Switch>
                <Suspense fallback={null}>
                    <Route path='/oview' component={Overview} />
                    <Route path='/detail' component={TrialsDetail} />
                    <Route path='/' render={(): React.ReactNode => <Redirect to={{ pathname: '/oview' }} />} />
                </Suspense>
            </Switch>
        </App>
    </Router>,

    document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
