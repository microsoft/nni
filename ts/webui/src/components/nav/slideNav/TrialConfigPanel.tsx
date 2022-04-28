import * as React from 'react';
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

interface LogPanelState {
    panelInnerHeight: number;
    innerWidth: number;
}

/**
 * search space
 * config
 * model
 */

class TrialConfigPanel extends React.Component<LogPanelProps, LogPanelState> {
    constructor(props: LogPanelProps) {
        super(props);

        this.state = {
            panelInnerHeight: window.innerHeight,
            innerWidth: window.innerWidth
        };
    }

    // use arrow function for change window size met error: this.setState is not a function
    setLogPanelHeight = (): void => {
        this.setState(() => ({ panelInnerHeight: window.innerHeight, innerWidth: window.innerWidth }));
    };

    async componentDidMount(): Promise<void> {
        window.addEventListener('resize', this.setLogPanelHeight);
    }

    componentWillUnmount(): void {
        window.removeEventListener('resize', this.setLogPanelHeight);
    }

    render(): React.ReactNode {
        const { hideConfigPanel, panelName } = this.props;
        const { panelInnerHeight, innerWidth } = this.state;
        const monacoEditorHeight = caclMonacoEditorHeight(panelInnerHeight);
        const blacklist = [
            'id',
            'logDir',
            'startTime',
            'endTime',
            'experimentName',
            'searchSpace',
            'trainingServicePlatform'
        ];
        const filter = (key: string, val: any): any => {
            return blacklist.includes(key) ? undefined : val;
        };
        const profile = lodash.cloneDeep(EXPERIMENT.profile);
        profile.execDuration = convertDuration(profile.execDuration);

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
    }
}

export default TrialConfigPanel;
