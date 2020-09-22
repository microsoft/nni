import React, { useState, useCallback } from 'react';
import { DefaultButton, Stack } from '@fluentui/react';
import TrialConfigPanel from './TrialConfigPanel';
import '../../../static/style/overview/config.scss'

export const TrialConfigButton = (): any => {
    const [isShowConfigPanel, setShowConfigPanle] = useState(false);
    const [activeTab, setActiveTab] = useState('1');
    const hideConfigPanel = useCallback(() => setShowConfigPanle(false), []);
    const showTrialConfigpPanel = useCallback(() => {
        setShowConfigPanle(true);
        setActiveTab('config');
    }, []);
    const showSearchSpacePanel = useCallback(() => {
        setShowConfigPanle(true);
        setActiveTab('search space');
    }, []);
    return (
        <React.Fragment>
            <Stack className="config" >
                <DefaultButton
                    text='Config'
                    onClick={showTrialConfigpPanel}
                />
                <DefaultButton
                    text='Search space'
                    onClick={showSearchSpacePanel}
                />
            </Stack>
            {isShowConfigPanel && <TrialConfigPanel hideConfigPanel={hideConfigPanel} activeTab={activeTab} />}
        </React.Fragment>
    );
};

