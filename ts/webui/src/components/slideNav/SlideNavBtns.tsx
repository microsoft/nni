import React, { useState, useCallback } from 'react';
import { Stack } from '@fluentui/react';
import TrialConfigPanel from './TrialConfigPanel';
import LogPanel from '../modals/LogPanel';
import IconButtonTemplate from './IconButtonTemplet';
import '../../static/style/overview/panel.scss';

export const SlideNavBtns = (): any => {
    const [isShowConfigPanel, setShowConfigPanle] = useState(false);
    const [isShowLogPanel, setShowLogPanel] = useState(false);
    const [panelName, setPanelName] = useState('');
    const hideConfigPanel = useCallback(() => setShowConfigPanle(false), []);
    const showTrialConfigpPanel = useCallback(() => {
        setShowConfigPanle(true);
        setPanelName('config');
    }, []);
    const showSearchSpacePanel = useCallback(() => {
        setShowConfigPanle(true);
        setPanelName('search space');
    }, []);
    const showLogPanel = useCallback(() => {
        setShowLogPanel(true);
    }, []);
    const hideLogPanel = useCallback(() => {
        setShowLogPanel(false);
    }, []);
    return (
        // right side nav buttons
        <React.Fragment>
            <Stack className='config'>
                <IconButtonTemplate icon='DocumentSearch' btuName='Search space' event={showSearchSpacePanel} />
                <IconButtonTemplate icon='Archive' btuName='Config' event={showTrialConfigpPanel} />
                <IconButtonTemplate icon='FilePDB' btuName='Log files' event={showLogPanel} />
            </Stack>
            {isShowConfigPanel && <TrialConfigPanel panelName={panelName} hideConfigPanel={hideConfigPanel} />}
            {/* the panel for dispatcher & nnimanager log message */}
            {isShowLogPanel && <LogPanel closeDrawer={hideLogPanel} />}
        </React.Fragment>
    );
};
