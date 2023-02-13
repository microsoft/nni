import React, { useState, useEffect } from 'react';
import { getPrefix } from '@static/function';
import { TOOLTIPSTYLE } from '@static/const';
import { DirectionalHint, TooltipHost } from '@fluentui/react';

/**
 *
 * for nav bar: document, github, nni-version tips
 */

interface TooltipHostIndexProps {
    iconName: string;
    tooltip: string;
    directional: string;
    iconClickEvent: () => void;
}

const LinksIcon = (props: TooltipHostIndexProps): any => {
    const { iconName, tooltip, iconClickEvent, directional } = props;
    const [mouHover, setHover] = useState(false);
    const [imgSrc, setImgsrc] = useState(`${getPrefix() || ''}/icons/${iconName}.png`);
    useEffect(() => {
        if (mouHover === true) {
            setImgsrc(`${getPrefix() || ''}/icons/${iconName}-1.png`);
        } else {
            setImgsrc(`${getPrefix() || ''}/icons/${iconName}.png`);
        }
    }, [mouHover]);
    return (
        <div>
            <TooltipHost
                content={tooltip}
                directionalHint={directional === 'right' ? DirectionalHint.rightCenter : DirectionalHint.bottomCenter}
                tooltipProps={TOOLTIPSTYLE}
            >
                <div
                    className='cursor'
                    onMouseEnter={(): void => setHover(true)}
                    onMouseLeave={(): void => setHover(false)}
                >
                    <img className='icon' onClick={iconClickEvent} src={imgSrc} alt={iconName} />
                </div>
            </TooltipHost>
        </div>
    );
};

export default LinksIcon;
