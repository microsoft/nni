import 'babel-polyfill';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import App from './App';
import { Router, Route, browserHistory, IndexRedirect } from 'react-router';
import registerServiceWorker from './registerServiceWorker';
import Accuracy from './components/Accuracy';
import Para from './components/Para';
import TrialStatus from './components/TrialStatus';
import Tensor from './components/Tensor';
import Control from './components/Control';
import Sessionpro from './components/Sessionpro';
import './index.css';

ReactDOM.render(
    <Router history={browserHistory}>
        <Route path="/" component={App}>
            <IndexRedirect to="/oview" />
            <Route path="/oview" component={Sessionpro} />
            <Route path="/hyper" component={Para} />
            <Route path="/trastaus" component={TrialStatus} />
            <Route path="/tensor" component={Tensor} />
            <Route path="/control" component={Control} />
            <Route path="/all" component={Accuracy} />
        </Route>
    </Router>,
    document.getElementById('root') as HTMLElement
);
registerServiceWorker();
