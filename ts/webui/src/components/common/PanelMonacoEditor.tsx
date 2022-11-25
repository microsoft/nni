import React, { useState, useEffect } from 'react';
import { Stack, Panel, PrimaryButton } from '@fluentui/react';
import MonacoEditor from 'react-monaco-editor';
import { caclMonacoEditorHeight } from '@static/function';
import '@style/logPanel.scss';

interface LogPanelProps {
    hideConfigPanel: () => void;
    panelName: string;
    panelContent: string;
}

/**
 * search space
 * config
 * retiarii parameter
 * panel
 */
const PanelMonacoEditor = (props: LogPanelProps): any => {
    const [panelInnerHeight, setPanelInnerHeight] = useState(window.innerHeight as number);
    // use arrow function for change window size met error: this.setState is not a function
    const setLogPanelHeight = (): void => {
        setPanelInnerHeight(window.innerHeight);
    };
    useEffect(() => {
        window.addEventListener('resize', setLogPanelHeight);
        return window.removeEventListener('resize', setLogPanelHeight); // return function === componentWillUnmount
    }, []); // [] === componentDidMount

    const { hideConfigPanel, panelName, panelContent } = props;
    const monacoEditorHeight = caclMonacoEditorHeight(panelInnerHeight);

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
                    <div>
                        <div className='panelName'>{panelName}</div>
                        <MonacoEditor
                            height={monacoEditorHeight}
                            language='json'
                            theme='vs-light'
                            value={panelContent}
                            options={{
                                minimap: { enabled: false },
                                readOnly: true,
                                automaticLayout: true,
                                wordWrap: 'on'
                            }}
                        />
                    </div>
                    <PrimaryButton text='Close' className='configClose' onClick={hideConfigPanel} />
                </div>
            </Panel>
        </Stack>
    );
};

export default PanelMonacoEditor;
