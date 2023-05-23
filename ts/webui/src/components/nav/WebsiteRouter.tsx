import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { getPrefix } from '@static/function';
import { TOOLTIPSTYLE } from '@static/const';
import { DirectionalHint, TooltipHost } from '@fluentui/react';

// feedback, document, version btns
type pagesType = 'Overview' | 'Trials detail';
interface WebRoutersInterface {
    currentPage: pagesType;
    changeCurrentPage: (value: pagesType) => void;
}
const WebRouters = (props: WebRoutersInterface): any => {
    const { currentPage, changeCurrentPage } = props; // Overview or Trials detail
    const [overviewImgsrc, setOverviewImgsrc] = useState(
        // window.location.pathname === '/oview'
        currentPage === 'Overview'
            ? `${getPrefix() || ''}/icons/overview-1.png`
            : `${getPrefix() || ''}/icons/overivew.png`
    );
    const [detailImgsrc, setdetailImgsrc] = useState(
        // window.location.pathname === '/detail'
        currentPage === 'Trials detail'
            ? `${getPrefix() || ''}/icons/detail-1.png`
            : `${getPrefix() || ''}/icons/detail.png`
    );
    const [overviewMouhover, setOverviewMouhover] = useState(false);
    const [detailMouhover, setDetailMouhover] = useState(false);
    useEffect(() => {
        if (
            overviewMouhover === false &&
            !(window.location.pathname.endsWith('/oview') || window.location.pathname.endsWith('/'))
        ) {
            setOverviewImgsrc(`${getPrefix() || ''}/icons/overview.png`);
        } else {
            setOverviewImgsrc(`${getPrefix() || ''}/icons/overview-1.png`);
        }
        if (detailMouhover === false && !window.location.pathname.endsWith('/detail')) {
            setdetailImgsrc(`${getPrefix() || ''}/icons/detail.png`);
        } else {
            setdetailImgsrc(`${getPrefix() || ''}/icons/detail-1.png`);
        }
    }, [overviewMouhover, detailMouhover]);

    return (
        <div>
            <TooltipHost
                content='Overview'
                directionalHint={DirectionalHint.rightCenter}
                tooltipProps={TOOLTIPSTYLE}
                className='tooltip-main-icon'
            >
                <NavLink to='/oview'>
                    <div
                        className='icon'
                        onMouseEnter={() => setOverviewMouhover(true)}
                        onMouseLeave={() => setOverviewMouhover(false)}
                        onClick={() => {
                            setOverviewImgsrc(`${getPrefix() || ''}/icons/overview-1.png`);
                            setdetailImgsrc(`${getPrefix() || ''}/icons/detail.png`);
                            changeCurrentPage('Overview');
                        }}
                    >
                        <img src={overviewImgsrc} alt='Overview icon' />
                    </div>
                </NavLink>
            </TooltipHost>
            <TooltipHost
                content='Trials detail'
                directionalHint={DirectionalHint.rightCenter}
                tooltipProps={TOOLTIPSTYLE}
                className='tooltip-main-icon'
            >
                <NavLink to='/detail'>
                    <div
                        className='icon'
                        onMouseEnter={() => setDetailMouhover(true)}
                        onMouseLeave={() => setDetailMouhover(false)}
                        onClick={() => {
                            setdetailImgsrc(`${getPrefix() || ''}/icons/detail-1.png`);
                            setOverviewImgsrc(`${getPrefix() || ''}/icons/overview.png`);
                            changeCurrentPage('Trials detail');
                        }}
                    >
                        <img src={detailImgsrc} alt='Details icon' />
                    </div>
                </NavLink>
            </TooltipHost>
        </div>
    );
};

export default WebRouters;
