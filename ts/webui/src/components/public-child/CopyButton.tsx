import React, { useRef } from 'react';
import PropTypes from 'prop-types';
import copy from 'copy-to-clipboard';
import { IconButton, FontSizes, TooltipHost } from '@fluentui/react';
import { TOOLTIP_BACKGROUND_COLOR } from '../../static/const';

const COPIED_TOOLTIP_CLOSE_DELAY = 1000;

const CopyButton = ({ value, hideTooltip }): any => {
    const ref = useRef(null);
    return (
        <div>
            <IconButton
                iconProps={{ iconName: 'Copy' }}
                styles={{ icon: [{ fontSize: FontSizes.small }] }}
                onClick={(event: React.SyntheticEvent<EventTarget>): void => {
                    event.preventDefault();
                    copy(value);
                    ref.current && (ref as any).current.show();
                    setTimeout(() => {
                        ref.current !== null && (ref as any).current.dismiss();
                    }, COPIED_TOOLTIP_CLOSE_DELAY);
                }}
            />
            <TooltipHost
                hidden={hideTooltip}
                content='Copied'
                componentRef={ref}
                delay={0}
                tooltipProps={{
                    calloutProps: {
                        styles: {
                            beak: { background: TOOLTIP_BACKGROUND_COLOR },
                            beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
                            calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
                        }
                    }
                }}
            />
        </div>
    );
};

CopyButton.propTypes = {
    value: PropTypes.string.isRequired,
    hideTooltip: PropTypes.bool
};

export default CopyButton;
