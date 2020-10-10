import * as React from 'react';
import { Stack, Panel, Pivot, PivotItem, PrimaryButton } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../../static/const';
import { convertDuration } from '../../../static/function';
import { prettyStringify } from '../../../static/json_util';
import lodash from 'lodash';
import '../../../static/style/logDrawer.scss';

interface LogDrawerProps {
    hideConfigPanel: () => void;
    activeTab?: string;
}

interface LogDrawerState {
    panelInnerHeight: number;
}

class TrialConfigPanel extends React.Component<LogDrawerProps, LogDrawerState> {
    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            panelInnerHeight: window.innerHeight
        };
    }

    setLogDrawerHeight(): void {
        this.setState(() => ({ panelInnerHeight: window.innerHeight }));
    }

    async componentDidMount(): Promise<void> {
        window.addEventListener('resize', this.setLogDrawerHeight);
    }

    componentWillUnmount(): void {
        window.removeEventListener('resize', this.setLogDrawerHeight);
    }

    render(): React.ReactNode {
        const { hideConfigPanel, activeTab } = this.props;
        const { panelInnerHeight } = this.state;
        // [marginTop 16px] + [Search space 46px] +
        // button[height: 32px, marginTop: 45px, marginBottom: 25px] + [padding-bottom: 20px]
        const monacoEditorHeight = panelInnerHeight - 184;
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
        profile.params.maxExecDuration = convertDuration(profile.params.maxExecDuration);
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
                    <div className='log-tab-body'>
                        <Pivot initialSelectedKey={activeTab} style={{ minHeight: 190, paddingTop: '16px' }}>
                            <PivotItem headerText='Search space' itemKey='search space'>
                                <MonacoEditor
                                    height={monacoEditorHeight}
                                    language='json'
                                    theme='vs-light'
                                    value={prettyStringify(EXPERIMENT.searchSpace, 300, 2)}
                                    options={MONACO}
                                />
                            </PivotItem>
                            <PivotItem headerText='Config' itemKey='config'>
                                <div className='profile'>
                                    <MonacoEditor
                                        width='100%'
                                        height={monacoEditorHeight}
                                        language='json'
                                        theme='vs-light'
                                        value={showProfile}
                                        options={MONACO}
                                    />
                                </div>
                            </PivotItem>
                        </Pivot>
                    </div>
                    <PrimaryButton text='Close' className='configClose' onClick={hideConfigPanel} />
                </Panel>
            </Stack>
        );
    }
}

export default TrialConfigPanel;
