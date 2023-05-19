import * as React from 'react';
import { Stack } from '@fluentui/react';
import { expandTrialGap } from '@components/common/Gap';

interface TrialLogProps {
    logStr: string;
    logName: string;
}

const TrialLog = (props: TrialLogProps): any => {
    const { logStr, logName } = props;
    const isHyperlink = logStr.toLowerCase().startsWith('http');

    return (
        <Stack horizontal className='logpath' tokens={expandTrialGap}>
            <span className='logName'>{logName}</span>
            {isHyperlink ? (
                <a className='link' rel='noopener noreferrer' href={logStr} target='_blank'>
                    {logStr}
                </a>
            ) : (
                <span className='fontColor333'>{logStr}</span>
            )}
        </Stack>
    );
};

export default TrialLog;
