import * as React from 'react';
import { NavLink } from 'react-router-dom';

import { getPrefix } from '@static/function';
const activeClassName = 'selected';

const OVERVIEWTABS = (
    <NavLink to='/oview' className={({ isActive }) => (isActive ? `${activeClassName} link` : 'link')}>
        <span className='common-tabs'>Overview</span>
    </NavLink>
);

const DETAILTABS = (
    <NavLink to='/detail' className={({ isActive }) => (isActive ? `${activeClassName} link` : 'link')}>
        <span className='common-tabs'>Trials detail</span>
    </NavLink>
);

const NNILOGO = (
    <NavLink to='/oview'>
        <img src={(getPrefix() || '') + '/logo.png'} alt='NNI logo' style={{ height: 40 }} />
    </NavLink>
);

export { OVERVIEWTABS, DETAILTABS, NNILOGO };
