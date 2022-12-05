import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { getPrefix } from '@static/function';
import { TOOLTIPSTYLE } from '@static/const';
import { DirectionalHint, TooltipHost } from '@fluentui/react';

// feedback, document, version btns

interface TooltipHostIndexProps {
    iconName: string;
    tooltip: string;
    pageURL: string;
}

const TooltipHostForIcon = (props: TooltipHostIndexProps): any => {
    // const { updateOverviewPage } = useContext(AppContext);
    const { iconName, tooltip, pageURL } = props;
    const [isActivePage, setIsActivePage] = useState(window.location.pathname === pageURL);
    const [overview, setOverview] = useState(
        isActivePage ? `${getPrefix() || ''}/icons/${iconName}-1.png` : `${getPrefix() || ''}/icons/${iconName}.png`
    );
    const [mouHover, setMouhover] = useState(false);
    useEffect(() => {
        // 方法简写 false && 不等，普通，其他高亮
        if (mouHover === true) {
            setOverview(`${getPrefix() || ''}/icons/${iconName}-1.png`);
        } else {
            if (window.location.pathname === pageURL) {
                setOverview(`${getPrefix() || ''}/icons/${iconName}-1.png`);
            } else {
                setOverview(`${getPrefix() || ''}/icons/${iconName}.png`);
            }
        }
    }, [mouHover, isActivePage]);
    /***
     * 1. mouHover 进去高亮，离开变灰 done
     * 2. 这个组件可以扩展为多tab
     * 3. 目前没打破的瓶颈：选中其他页面时，要赋值之前选中页面为普通模式。找不到之前的元素。。。
     */
    return (
        <div>
            <TooltipHost
                content={tooltip}
                directionalHint={DirectionalHint.rightCenter}
                tooltipProps={TOOLTIPSTYLE}
                className='tooltip-main-icon'
            >
                <NavLink to={pageURL}>
                    <div
                        className='icon'
                        onMouseEnter={() => setMouhover(true)}
                        onMouseLeave={() => setMouhover(false)}
                        onClick={() => {
                            setIsActivePage(window.location.pathname === pageURL);
                            setOverview(`${getPrefix() || ''}/icons/${iconName}-1.png`);
                            // updateOverviewPage();
                        }}
                    >
                        <img src={overview} alt={iconName.toString()} />
                    </div>
                </NavLink>
            </TooltipHost>
        </div>
    );
};

export default TooltipHostForIcon;
