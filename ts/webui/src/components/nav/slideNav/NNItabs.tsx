import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { getPrefix } from '@static/function';
const activeClassName = 'selected';
const OVERVIEWTABS = (
    <NavLink to='/oview' className={({ isActive }) => (isActive ? `${activeClassName} link` : 'link')}>
        <span className='common-tabs'>Overview</span>
    </NavLink>
);

const DETAILTABS = (
    <a href='/detail' className='link'>
        <span className='common-tabs'>Trials detail</span>
    </a>
);

const NNILOGO = (
    <NavLink to='/oview'>
        <img src={(getPrefix() || '') + '/logo.png'} alt='NNI logo' style={{ height: 40 }} />
    </NavLink>
);

export const OVERVIEWTABSNew = () => {
    const [overview, setOverview] = useState(`${getPrefix() || ''}/icons/overview.png`);
    return (
        <NavLink to='/oview' className={({ isActive }) => (isActive ? `${activeClassName} link` : 'link')}>
            <div
                className='icon'
                onClick={() => {
                    setOverview(`${getPrefix() || ''}/icons/overview-1.png`);
                }}
            >
                {/* <img src={(getPrefix() || '') + '/icons/overview.png'} /> */}
                <img src={overview} alt='overview' />
            </div>
        </NavLink>
    );
};

export { OVERVIEWTABS, DETAILTABS, NNILOGO };
