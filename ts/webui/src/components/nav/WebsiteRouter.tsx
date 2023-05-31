import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { getPrefix } from '@static/function';
import { TOOLTIPSTYLE } from '@static/const';
import { DirectionalHint, TooltipHost } from '@fluentui/react';
import { getRouter } from './Nav';

// feedback, document, version btns
type pagesType = 'Overview' | 'Trials detail';
interface WebRoutersInterface {
    currentPage: pagesType;
    changeCurrentPage: (value: pagesType) => void;
}

const WebRouters = (props: WebRoutersInterface): any => {
    const { currentPage, changeCurrentPage } = props; // Overview or Trials detail
    const prefix = getPrefix() || '';
    const [overviewImgsrc, setOverviewImgsrc] = useState(
        currentPage === 'Overview' ? `${prefix}/icons/overview-1.png` : `${prefix}/icons/overivew.png`
    );
    const [detailImgsrc, setdetailImgsrc] = useState(
        currentPage === 'Trials detail' ? `${prefix}/icons/detail-1.png` : `${prefix}/icons/detail.png`
    );
    const [overviewMouhover, setOverviewMouhover] = useState(false);
    const [detailMouhover, setDetailMouhover] = useState(false);
    useEffect(() => {
        const result = getRouter();
        if (overviewMouhover === false && !(result === '/oview')) {
            setOverviewImgsrc(`${prefix}/icons/overview.png`);
        } else {
            setOverviewImgsrc(`${prefix}/icons/overview-1.png`);
        }
        if (detailMouhover === false && !(result === '/detail')) {
            setdetailImgsrc(`${prefix}/icons/detail.png`);
        } else {
            setdetailImgsrc(`${prefix}/icons/detail-1.png`);
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
                            setOverviewImgsrc(`${prefix}/icons/overview-1.png`);
                            setdetailImgsrc(`${prefix}/icons/detail.png`);
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
                            setdetailImgsrc(`${prefix}/icons/detail-1.png`);
                            setOverviewImgsrc(`${prefix}/icons/overview.png`);
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
