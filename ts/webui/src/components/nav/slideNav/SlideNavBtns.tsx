import React, { useState, useCallback } from 'react';
import TrialConfigPanel from './TrialConfigPanel';
import LogPanel from './LogPanel';
import LinksIcon from '@components/nav/LinksIcon';
import '@style/nav/slideNavBtns.scss';

/***
 * this file is the container of [config, search space, dispatcher/nnimanager log]
 * these three button is in the right of page
 */

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
        <React.Fragment>
            <LinksIcon
                tooltip='Search-space'
                iconName='search-space'
                directional='bottom'
                iconClickEvent={showSearchSpacePanel}
            />
            <LinksIcon tooltip='Config' iconName='config' directional='bottom' iconClickEvent={showTrialConfigpPanel} />
            <LinksIcon tooltip='Log files' iconName='log' directional='bottom' iconClickEvent={showLogPanel} />
            {isShowConfigPanel && <TrialConfigPanel panelName={panelName} hideConfigPanel={hideConfigPanel} />}
            {/* the panel for dispatcher & nnimanager log message */}
            {isShowLogPanel && <LogPanel closePanel={hideLogPanel} />}
        </React.Fragment>
    );
};
