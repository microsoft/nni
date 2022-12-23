import React, { useState, useCallback } from 'react';
import { Stack } from '@fluentui/react/lib/Stack';
import { DefaultButton } from '@fluentui/react/lib/Button';
import LogPanel from '@components/nav/slideNav/LogPanel';
import { Important, Cancel } from '@components/fluent/Icon';
import { EXPERIMENT } from '@static/datamodel';
import '@style/nav/slideNavBtns.scss';

// This file is for <experiment error> message model
// Position: show the message in the lower right corner of the page
export const ErrorMessage: React.FunctionComponent = () => {
    const [hideDialog, setHideDialog] = useState(EXPERIMENT.status === 'ERROR' ? true : false);
    const [isShowLogPanel, setShowLogPanel] = useState(false);
    const closeLogPanel = useCallback(() => setShowLogPanel(false), []);
    const ShowLogPanel = useCallback(() => setShowLogPanel(true), []);

    return (
        <>
            {hideDialog ? (
                <div className='experiment-error'>
                    <Stack horizontal className='head'>
                        <Stack horizontal>
                            <div className='icon'>{Important}</div>
                            <div className='title'>Error</div>
                        </Stack>
                        <Stack className='close cursor' onClick={() => setHideDialog(false)}>
                            {Cancel}
                        </Stack>
                    </Stack>
                    <div className='message'>{EXPERIMENT.error}</div>
                    <Stack horizontalAlign='end' className='experiment-error-buttons'>
                        <DefaultButton className='detailsBtn' onClick={ShowLogPanel} text='Learn more' />
                    </Stack>
                    {/* learn about click -> default active key is dispatcher. */}
                    {isShowLogPanel ? <LogPanel closePanel={closeLogPanel} activeTab='dispatcher' /> : null}
                </div>
            ) : null}
        </>
    );
};
