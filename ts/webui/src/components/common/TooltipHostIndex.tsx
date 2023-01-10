import * as React from 'react';
import { TOOLTIPSTYLE } from '@static/const';
import { DirectionalHint, TooltipHost } from '@fluentui/react';

interface TooltipHostIndexProps {
    value: string;
}

const TooltipHostIndex = (props: TooltipHostIndexProps): any => {
    const { value } = props;
    const length = String(value).length;
    return (
        <>
            {length >= 15 ? (
                <div>
                    <TooltipHost
                        content={value}
                        directionalHint={DirectionalHint.bottomLeftEdge}
                        tooltipProps={TOOLTIPSTYLE}
                    >
                        <div className='ellipsis name'>{value}</div>
                    </TooltipHost>
                </div>
            ) : (
                <div className='name'>{value}</div>
            )}
        </>
    );
};

export default TooltipHostIndex;
