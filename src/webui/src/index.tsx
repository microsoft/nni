import 'babel-polyfill';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import App from './App';
import { Router, Route, browserHistory, IndexRedirect } from 'react-router';
import registerServiceWorker from './registerServiceWorker';
import Tensor from './components/Tensor';
import Control from './components/Control';
import Overview from './components/Overview';
import TrialsDetail from './components/TrialsDetail';
// import TrialsDetail from './components/TrialsDetail';
import './index.css';

ReactDOM.render(
    <Router history={browserHistory}>
        <Route path="/" component={App}>
            <IndexRedirect to="/oview" />
            <Route path="/oview" component={Overview} />
            <Route path="/detail" component={TrialsDetail} />
            <Route path="/tensor" component={Tensor} />
            <Route path="/control" component={Control} />
        </Route>
    </Router>,
    document.getElementById('root') as HTMLElement
);
registerServiceWorker();
