import * as React from 'react';
import { Stack, Panel, Pivot, PivotItem, PrimaryButton } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { ConcurrencyContext } from '../../Overview';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../../static/const';
import '../../../static/style/logDrawer.scss';

interface LogDrawerProps {
    hideConfigPanel: () => void;
    activeTab?: string;
}

interface LogDrawerState {
    logDrawerHeight: number;
}

class TrialConfigPanel extends React.Component<LogDrawerProps, LogDrawerState> {
    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            logDrawerHeight: window.innerHeight - 48
        };
    }

    setLogDrawerHeight = (): void => {
        this.setState(() => ({ logDrawerHeight: window.innerHeight - 48 }));
    };

    async componentDidMount(): Promise<void> {
        window.addEventListener('resize', this.setLogDrawerHeight);
    }

    componentWillUnmount(): void {
        window.removeEventListener('resize', this.setLogDrawerHeight);
    }

    render(): React.ReactNode {
        const { hideConfigPanel, activeTab } = this.props;
        const { logDrawerHeight } = this.state;

        return (
            <Stack>
                <Panel
                    isOpen={true}
                    hasCloseButton={false}
                    isFooterAtBottom={true}
                    isLightDismiss={true}
                    onLightDismissClick={hideConfigPanel}
                >
                    <div className='log-tab-body'>
                        <Pivot initialSelectedKey={activeTab} style={{ minHeight: 190, paddingTop: '16px' }}>
                            <PivotItem headerText='Search space' itemKey="search space">
                                <MonacoEditor
                                    height={logDrawerHeight - 92 - 45}
                                    language='json'
                                    theme='vs-light'
                                    value={JSON.stringify(EXPERIMENT.searchSpace, null, 2)}
                                    options={MONACO}
                                />
                            </PivotItem>
                            <PivotItem headerText='Config' itemKey='config'>
                                <ConcurrencyContext.Consumer>
                                    {(value): React.ReactNode => {
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
                                            if (key === 'trialConcurrency') {
                                                return value.trialConcurrency;
                                            }
                                            return blacklist.includes(key) ? undefined : val;
                                        };
                                        const profile = JSON.stringify(EXPERIMENT.profile, filter, 2);

                                        return (
                                            <div className='profile'>
                                                <MonacoEditor
                                                    width='100%'
                                                    height={logDrawerHeight - 92 - 45}
                                                    language='json'
                                                    theme='vs-light'
                                                    value={profile}
                                                    options={MONACO}
                                                />
                                            </div>
                                        );
                                    }
                                    }
                                </ConcurrencyContext.Consumer>
                            </PivotItem>
                        </Pivot>
                    </div>
                    <PrimaryButton text='Close' className="configClose" onClick={hideConfigPanel} />
                </Panel>
            </Stack>
        );
    }
}

export default TrialConfigPanel;
