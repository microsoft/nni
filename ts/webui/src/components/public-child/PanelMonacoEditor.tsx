import * as React from 'react';
import { Stack, Panel, PrimaryButton } from '@fluentui/react';
import MonacoEditor from 'react-monaco-editor';
import { caclMonacoEditorHeight } from '../../static/function';
import '../../static/style/logDrawer.scss';

interface LogDrawerProps {
    hideConfigPanel: () => void;
    panelName: string;
    panelContent: string;
}

interface LogDrawerState {
    panelInnerHeight: number;
}

/**
 * search space
 * config
 * retiarii parameter
 * panel
 */

class PanelMonacoEditor extends React.Component<LogDrawerProps, LogDrawerState> {
    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            panelInnerHeight: window.innerHeight
        };
    }

    // use arrow function for change window size met error: this.setState is not a function
    setLogDrawerHeight = (): void => {
        this.setState(() => ({ panelInnerHeight: window.innerHeight, innerWidth: window.innerWidth }));
    };

    async componentDidMount(): Promise<void> {
        window.addEventListener('resize', this.setLogDrawerHeight);
    }

    componentWillUnmount(): void {
        window.removeEventListener('resize', this.setLogDrawerHeight);
    }

    render(): React.ReactNode {
        const { hideConfigPanel, panelName, panelContent } = this.props;
        const { panelInnerHeight } = this.state;
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
    }
}

export default PanelMonacoEditor;
