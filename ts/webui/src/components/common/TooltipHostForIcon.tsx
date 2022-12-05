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
    const { iconName, tooltip, pageURL } = props;
    const [overview, setOverview] = useState(`${getPrefix() || ''}/icons/${iconName}.png`);
    const [mouHover, setMouhover] = useState(false);
    useEffect(() => {
        if (mouHover === false) {
            setOverview(`${getPrefix() || ''}/icons/all-experiments.png`);
        } else {
            setOverview(`${getPrefix() || ''}/icons/all-experiments-1.png`);
        }
    }, [mouHover]);

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
                    >
                        <img src={overview} alt={iconName} />
                    </div>
                </NavLink>
            </TooltipHost>
        </div>
    );
};

export default TooltipHostForIcon;
