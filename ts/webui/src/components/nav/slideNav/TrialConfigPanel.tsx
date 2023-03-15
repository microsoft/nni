import React, { useState, useCallback, useEffect } from 'react';
import { Stack, Panel, PrimaryButton } from '@fluentui/react';
import lodash from 'lodash';
import MonacoEditor from 'react-monaco-editor';
import { EXPERIMENT } from '@static/datamodel';
import { MONACO } from '@static/const';
import { convertDuration, caclMonacoEditorHeight } from '@static/function';
import { prettyStringify } from '@static/jsonutil';
import '@static/style/logPanel.scss';

interface LogPanelProps {
    hideConfigPanel: () => void;
    panelName: string;
}

/**
 * search space
 * config
 * model
 */

// init const
const blacklist = ['id', 'logDir', 'startTime', 'endTime', 'experimentName', 'searchSpace', 'trainingServicePlatform'];

const TrialConfigPanel = (props: LogPanelProps): any => {
    const [panelInnerHeight, setPanelInnerHeight] = useState(window.innerHeight as number);
    const [innerWidth, setInnerWidth] = useState(window.innerWidth as number);

    // use arrow function for change window size met error: this.setState is not a function
    const setLogPanelHeight = useCallback(() => {
        setPanelInnerHeight(window.innerHeight);
        setInnerWidth(window.innerWidth);
    }, []);

    useEffect(() => {
        window.addEventListener('resize', setLogPanelHeight);
        return function () {
            window.removeEventListener('resize', setLogPanelHeight);
        };
    }, []);

    const { hideConfigPanel, panelName } = props;
    const monacoEditorHeight = caclMonacoEditorHeight(panelInnerHeight);
    const filter = (key: string, val: any): any => {
        return blacklist.includes(key) ? undefined : val;
    };
    const profile = lodash.cloneDeep(EXPERIMENT.profile);
    profile.execDuration = convertDuration(profile.execDuration) as any; // FIXME
    const prettyWidth = innerWidth > 1400 ? 100 : 60;
    const showProfile = JSON.stringify(profile, filter, 2);

    return (
        <Stack>
            <Panel
                isOpen={true}
                hasCloseButton={false}
                isFooterAtBottom={true}
                isLightDismiss={true}
                onLightDismissClick={hideConfigPanel}
            >
                <div className='panel'>
                    {panelName === 'search space' ? (
                        <div>
                            <div className='panelName'>Search space</div>
                            <MonacoEditor
                                height={monacoEditorHeight}
                                language='json'
                                theme='vs-light'
                                value={prettyStringify(EXPERIMENT.searchSpace, prettyWidth, 2)}
                                options={MONACO}
                            />
                        </div>
                    ) : (
                        <div className='profile'>
                            <div className='panelName'>Config</div>
                            <MonacoEditor
                                width='100%'
                                height={monacoEditorHeight}
                                language='json'
                                theme='vs-light'
                                value={showProfile}
                                options={MONACO}
                            />
                        </div>
                    )}
                    <PrimaryButton text='Close' className='configClose' onClick={hideConfigPanel} />
                </div>
            </Panel>
        </Stack>
    );
};

export default TrialConfigPanel;
