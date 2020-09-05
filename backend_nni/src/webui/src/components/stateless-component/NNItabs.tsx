import * as React from 'react';
import { Link } from 'react-router';

const OVERVIEWTABS = (
    <Link to={'/oview'} activeClassName="high-light" className="common-tabs">
        Overview
    </Link>
);

const DETAILTABS = (
    <Link to={'/detail'} activeClassName="high-light" className="common-tabs">
        Trials detail
    </Link>
);

const NNILOGO = (
    <Link to={'/oview'}>
        <img
            src={require('../../static/img/logo2.png')}
            alt="NNI logo"
            style={{height: 40}}
        />
    </Link>
);

export { OVERVIEWTABS, DETAILTABS, NNILOGO };