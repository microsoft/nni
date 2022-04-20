import React, { useState, useCallback } from 'react';
import { Stack, DefaultButton, Icon } from '@fluentui/react';
import MediaQuery from 'react-responsive';
import TrialConfigPanel from './TrialConfigPanel';
import LogPanel from './LogPanel';
import IconButtonTemplate from './IconButtonTemplet';
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
        // right side nav buttons
        <React.Fragment>
            <Stack className='config'>
                <MediaQuery maxWidth={1799}>
                    <IconButtonTemplate icon='DocumentSearch' btuName='Search space' event={showSearchSpacePanel} />
                    <IconButtonTemplate icon='Archive' btuName='Config' event={showTrialConfigpPanel} />
                    <IconButtonTemplate icon='FilePDB' btuName='Log files' event={showLogPanel} />
                </MediaQuery>
                <MediaQuery minWidth={1798}>
                    <div className='container'>
                        <DefaultButton onClick={showSearchSpacePanel} className='maxScrBtn'>
                            <Icon iconName='DocumentSearch' />
                            <span className='margin'>Search space</span>
                        </DefaultButton>
                    </div>
                    <div className='container'>
                        <DefaultButton onClick={showTrialConfigpPanel} className='maxScrBtn configBtn'>
                            <Icon iconName='Archive' />
                            <span className='margin'>Config</span>
                        </DefaultButton>
                    </div>
                    <div className='container'>
                        <DefaultButton onClick={showLogPanel} className='maxScrBtn logBtn'>
                            <Icon iconName='FilePDB' />
                            <span className='margin'>Log files</span>
                        </DefaultButton>
                    </div>
                </MediaQuery>
            </Stack>
            {isShowConfigPanel && <TrialConfigPanel panelName={panelName} hideConfigPanel={hideConfigPanel} />}
            {/* the panel for dispatcher & nnimanager log message */}
            {isShowLogPanel && <LogPanel closePanel={hideLogPanel} />}
        </React.Fragment>
    );
};
